//! Lucene's document API (`org.apache.lucene.document`): a [`Document`] is a
//! list of [`IndexableField`]s, each carrying its own [`FieldType`], and the
//! writer decides nothing -- it inverts, stores, doc-values and points each
//! field exactly as its type says (`IndexingChain.processDocument`).
//!
//! This is the shape Java callers write, and the one every typed field of the
//! package ([`IntField`], [`KeywordField`], [`FeatureField`], the `*Range`
//! fields, ...) is a constructor of. [`IndexWriter::add_fields_documents`]
//! turns each document into the writer's pre-decided
//! [`ExplicitDocument`] -- the output of
//! `IndexingChain` for that document: fields registered in first-appearance
//! order (`FieldInfos.FieldNumbers`), each value inverted through its own
//! token stream with the writer's analyzer, norms computed by the writer's
//! similarity, stored values in document order, doc values and packed points
//! one entry per value.
//!
//! # Rust shapes
//!
//! - [`IndexableField`] is Java's interface of the same name, one method per
//!   accessor `IndexingChain` reads; every typed field is a struct
//!   implementing it, as each is a `Field` subclass in Java.
//! - A `TokenStream` value is a [`FieldTokens`]: the token list plus the two
//!   end-of-stream attribute values, with bytes terms (a `BytesTermAttribute`
//!   stream) and a `TermFrequencyAttribute` per token, which
//!   `lucene_analysis::TokenStream` does not carry.
//! - A `Reader` value is a `String` ([`Field::from_reader`]): nothing on disk
//!   distinguishes the two, but a reader value is still refused as stored.
//! - `FieldType.freeze()` is kept: a frozen type refuses its setters.
//! - Exceptions are [`Error`]s, with Java's messages.
//!
//! Not supported by the writer this feeds (refused at registration): term
//! vectors and payloads.

mod date_tools;
mod doc_values;
mod feature;
mod geo;
mod indexing;
mod inet_address;
mod late_interaction;
mod numeric;
mod range;
mod shape;
mod shape_doc_values;
mod vectors;

pub mod column;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

use lucene_analysis::Analyzer;
pub use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
pub use lucene_codecs::stored_fields::FieldValue as StoredValue;

pub use date_tools::{DateTools, Resolution};
pub use doc_values::{
    BinaryDocValuesField, KeywordField, NumericDocValuesField, SortedDocValuesField,
    SortedNumericDocValuesField, SortedSetDocValuesField,
};
pub use feature::FeatureField;
pub use geo::{
    doc_value_high, doc_value_low, pack_doc_value, LatLonDocValuesField, LatLonPoint,
    XYDocValuesField, XYPointField,
};
pub use inet_address::InetAddressPoint;
pub use late_interaction::LateInteractionField;
pub use numeric::{
    double_to_sortable_long, float_to_sortable_int, int_to_sortable_bytes, long_to_sortable_bytes,
    sortable_bytes_to_int, sortable_bytes_to_long, sortable_double_bits, sortable_int_to_float,
    sortable_long_to_double, BinaryPoint, DoubleField, DoublePoint, FloatField, FloatPoint,
    IntField, IntPoint, LongField, LongPoint,
};
pub use range::{
    BinaryRangeDocValuesField, DoubleRange, DoubleRangeDocValuesField, FloatRange,
    FloatRangeDocValuesField, InetAddressRange, IntRange, IntRangeDocValuesField, LongRange,
    LongRangeDocValuesField, RangeQueryType,
};
pub use shape::{
    DecodedTriangle, LatLonShape, LatLonShapeDocValues, LatLonShapeDocValuesField, ShapeField,
    ShapeTriangle, TriangleType, XYShape, XYShapeDocValues, XYShapeDocValuesField, TRIANGLE_BYTES,
};
pub use shape_doc_values::{ShapeDocValues, ShapeEncoding};
pub use vectors::{KnnByteVectorField, KnnFloatVectorField};

#[cfg(doc)]
use crate::index_writer::ExplicitDocument;
#[cfg(doc)]
use crate::index_writer::IndexWriter;

/// Java's `IllegalArgumentException`/`IllegalStateException` from the
/// document API, with its message.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("{0}")]
    IllegalArgument(String),
    #[error("{0}")]
    IllegalState(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn illegal(message: impl Into<String>) -> Error {
    Error::IllegalArgument(message.into())
}

/// `IndexableField.numericValue()`'s `Number`, the four boxed types a field
/// holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
}

impl Number {
    /// `Number.longValue()`: a narrowing (saturating, `NaN` to `0`)
    /// conversion for the floating-point types, as Java's `(long)` cast.
    pub fn long_value(self) -> i64 {
        match self {
            Number::Int(v) => i64::from(v),
            Number::Long(v) => v,
            Number::Float(v) => v as i64,
            Number::Double(v) => v as i64,
        }
    }

    /// `Number.intValue()`.
    pub fn int_value(self) -> i32 {
        match self {
            Number::Int(v) => v,
            Number::Long(v) => v as i32,
            Number::Float(v) => v as i32,
            Number::Double(v) => v as i32,
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Int(v) => write!(f, "{v}"),
            Number::Long(v) => write!(f, "{v}"),
            Number::Float(v) => write!(f, "{v}"),
            Number::Double(v) => write!(f, "{v}"),
        }
    }
}

/// `InvertableType`: how the indexing chain turns an indexed field into
/// terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvertableType {
    /// The field's [`IndexableField::binary_value`] is its one term.
    Binary,
    /// The field produces a token stream ([`IndexableField::token_stream`]).
    TokenStream,
}

/// `Field.Store`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Store {
    Yes,
    No,
}

/// `PointValues.MAX_DIMENSIONS`.
pub const MAX_DIMENSIONS: i32 = 16;
/// `PointValues.MAX_INDEX_DIMENSIONS`.
pub const MAX_INDEX_DIMENSIONS: i32 = 8;
/// `PointValues.MAX_NUM_BYTES`.
pub const MAX_NUM_BYTES: i32 = 16;

/// `FieldType`: how one field instance is indexed.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldType {
    stored: bool,
    tokenized: bool,
    store_term_vectors: bool,
    store_term_vector_offsets: bool,
    store_term_vector_positions: bool,
    store_term_vector_payloads: bool,
    omit_norms: bool,
    index_options: IndexOptions,
    frozen: bool,
    doc_values_type: DocValuesType,
    doc_values_skip_index: DocValuesSkipIndexType,
    dimension_count: i32,
    index_dimension_count: i32,
    dimension_num_bytes: i32,
    vector_dimension: i32,
    vector_encoding: VectorEncoding,
    vector_similarity_function: VectorSimilarityFunction,
    attributes: Option<BTreeMap<String, String>>,
}

impl Default for FieldType {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! setter {
    ($(#[$m:meta])* $name:ident, $field:ident, $ty:ty) => {
        $(#[$m])*
        pub fn $name(&mut self, value: $ty) -> Result<()> {
            self.check_if_frozen()?;
            self.$field = value;
            Ok(())
        }
    };
}

impl FieldType {
    /// `new FieldType()`: not stored, tokenized, not indexed, nothing else.
    pub const fn new() -> Self {
        FieldType {
            stored: false,
            tokenized: true,
            store_term_vectors: false,
            store_term_vector_offsets: false,
            store_term_vector_positions: false,
            store_term_vector_payloads: false,
            omit_norms: false,
            index_options: IndexOptions::None,
            frozen: false,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index: DocValuesSkipIndexType::None,
            dimension_count: 0,
            index_dimension_count: 0,
            dimension_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
            attributes: None,
        }
    }

    /// `new FieldType(ref)`: an unfrozen copy.
    pub fn copy_of(other: &FieldType) -> Self {
        FieldType {
            frozen: false,
            ..other.clone()
        }
    }

    fn check_if_frozen(&self) -> Result<()> {
        if self.frozen {
            return Err(Error::IllegalState(
                "this FieldType is already frozen and cannot be changed".into(),
            ));
        }
        Ok(())
    }

    /// `freeze()`.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// The same type, frozen -- the builder form of `freeze()`.
    pub fn frozen(mut self) -> Self {
        self.frozen = true;
        self
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    pub fn stored(&self) -> bool {
        self.stored
    }
    pub fn tokenized(&self) -> bool {
        self.tokenized
    }
    pub fn store_term_vectors(&self) -> bool {
        self.store_term_vectors
    }
    pub fn store_term_vector_offsets(&self) -> bool {
        self.store_term_vector_offsets
    }
    pub fn store_term_vector_positions(&self) -> bool {
        self.store_term_vector_positions
    }
    pub fn store_term_vector_payloads(&self) -> bool {
        self.store_term_vector_payloads
    }
    pub fn omit_norms(&self) -> bool {
        self.omit_norms
    }
    pub fn index_options(&self) -> IndexOptions {
        self.index_options
    }
    pub fn doc_values_type(&self) -> DocValuesType {
        self.doc_values_type
    }
    pub fn doc_values_skip_index_type(&self) -> DocValuesSkipIndexType {
        self.doc_values_skip_index
    }
    pub fn point_dimension_count(&self) -> i32 {
        self.dimension_count
    }
    pub fn point_index_dimension_count(&self) -> i32 {
        self.index_dimension_count
    }
    pub fn point_num_bytes(&self) -> i32 {
        self.dimension_num_bytes
    }
    pub fn vector_dimension(&self) -> i32 {
        self.vector_dimension
    }
    pub fn vector_encoding(&self) -> VectorEncoding {
        self.vector_encoding
    }
    pub fn vector_similarity_function(&self) -> VectorSimilarityFunction {
        self.vector_similarity_function
    }
    /// `getAttributes()`: `None` until an attribute is put.
    pub fn attributes(&self) -> Option<&BTreeMap<String, String>> {
        self.attributes.as_ref()
    }

    setter!(set_stored, stored, bool);
    setter!(set_tokenized, tokenized, bool);
    setter!(set_store_term_vectors, store_term_vectors, bool);
    setter!(
        set_store_term_vector_offsets,
        store_term_vector_offsets,
        bool
    );
    setter!(
        set_store_term_vector_positions,
        store_term_vector_positions,
        bool
    );
    setter!(
        set_store_term_vector_payloads,
        store_term_vector_payloads,
        bool
    );
    setter!(set_omit_norms, omit_norms, bool);
    setter!(set_index_options, index_options, IndexOptions);
    setter!(set_doc_values_type, doc_values_type, DocValuesType);
    setter!(
        set_doc_values_skip_index_type,
        doc_values_skip_index,
        DocValuesSkipIndexType
    );

    /// `setDimensions(dimensionCount, dimensionNumBytes)`.
    pub fn set_dimensions(&mut self, dimension_count: i32, dimension_num_bytes: i32) -> Result<()> {
        self.set_dimensions_with_index(dimension_count, dimension_count, dimension_num_bytes)
    }

    /// `setDimensions(dimensionCount, indexDimensionCount, dimensionNumBytes)`.
    pub fn set_dimensions_with_index(
        &mut self,
        dimension_count: i32,
        index_dimension_count: i32,
        dimension_num_bytes: i32,
    ) -> Result<()> {
        self.check_if_frozen()?;
        if dimension_count < 0 {
            return Err(illegal(format!(
                "dimensionCount must be >= 0; got {dimension_count}"
            )));
        }
        if dimension_count > MAX_DIMENSIONS {
            return Err(illegal(format!(
                "dimensionCount must be <= {MAX_DIMENSIONS}; got {dimension_count}"
            )));
        }
        if index_dimension_count < 0 {
            return Err(illegal(format!(
                "indexDimensionCount must be >= 0; got {index_dimension_count}"
            )));
        }
        if index_dimension_count > dimension_count {
            return Err(illegal(format!(
                "indexDimensionCount must be <= dimensionCount: {dimension_count}; got \
                 {index_dimension_count}"
            )));
        }
        if index_dimension_count > MAX_INDEX_DIMENSIONS {
            return Err(illegal(format!(
                "indexDimensionCount must be <= {MAX_INDEX_DIMENSIONS}; got \
                 {index_dimension_count}"
            )));
        }
        if dimension_num_bytes < 0 {
            return Err(illegal(format!(
                "dimensionNumBytes must be >= 0; got {dimension_num_bytes}"
            )));
        }
        if dimension_num_bytes > MAX_NUM_BYTES {
            return Err(illegal(format!(
                "dimensionNumBytes must be <= {MAX_NUM_BYTES}; got {dimension_num_bytes}"
            )));
        }
        if dimension_count == 0 {
            if index_dimension_count != 0 {
                return Err(illegal(format!(
                    "when dimensionCount is 0, indexDimensionCount must be 0; got \
                     {index_dimension_count}"
                )));
            }
            if dimension_num_bytes != 0 {
                return Err(illegal(format!(
                    "when dimensionCount is 0, dimensionNumBytes must be 0; got \
                     {dimension_num_bytes}"
                )));
            }
        } else if index_dimension_count == 0 {
            return Err(illegal(format!(
                "when dimensionCount is > 0, indexDimensionCount must be > 0; got \
                 {index_dimension_count}"
            )));
        } else if dimension_num_bytes == 0 {
            return Err(illegal(format!(
                "when dimensionNumBytes is 0, dimensionCount must be 0; got {dimension_count}"
            )));
        }
        self.dimension_count = dimension_count;
        self.index_dimension_count = index_dimension_count;
        self.dimension_num_bytes = dimension_num_bytes;
        Ok(())
    }

    /// `setVectorAttributes(numDimensions, encoding, similarity)`.
    pub fn set_vector_attributes(
        &mut self,
        num_dimensions: i32,
        encoding: VectorEncoding,
        similarity: VectorSimilarityFunction,
    ) -> Result<()> {
        self.check_if_frozen()?;
        if num_dimensions <= 0 {
            return Err(illegal(format!(
                "vector numDimensions must be > 0; got {num_dimensions}"
            )));
        }
        self.vector_dimension = num_dimensions;
        self.vector_encoding = encoding;
        self.vector_similarity_function = similarity;
        Ok(())
    }

    /// `putAttribute(key, value)`: the previous value, if any.
    pub fn put_attribute(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Option<String>> {
        self.check_if_frozen()?;
        Ok(self
            .attributes
            .get_or_insert_with(BTreeMap::new)
            .insert(key.into(), value.into()))
    }
}

impl fmt::Display for FieldType {
    /// `FieldType.toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        if self.stored {
            parts.push("stored".into());
        }
        if self.index_options != IndexOptions::None {
            let mut s = String::from("indexed");
            if self.tokenized {
                s.push_str(",tokenized");
            }
            if self.store_term_vectors {
                s.push_str(",termVector");
            }
            if self.store_term_vector_offsets {
                s.push_str(",termVectorOffsets");
            }
            if self.store_term_vector_positions {
                s.push_str(",termVectorPosition");
            }
            if self.store_term_vector_payloads {
                s.push_str(",termVectorPayloads");
            }
            if self.omit_norms {
                s.push_str(",omitNorms");
            }
            if self.index_options != IndexOptions::DocsAndFreqsAndPositions {
                s.push_str(&format!(
                    ",indexOptions={}",
                    index_options_name(self.index_options)
                ));
            }
            parts.push(s);
        }
        if self.dimension_count != 0 {
            parts.push(format!(
                "pointDimensionCount={},pointIndexDimensionCount={},pointNumBytes={}",
                self.dimension_count, self.index_dimension_count, self.dimension_num_bytes
            ));
        }
        if self.doc_values_type != DocValuesType::None {
            parts.push(format!(
                "docValuesType={}",
                doc_values_type_name(self.doc_values_type)
            ));
        }
        f.write_str(&parts.join(","))
    }
}

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

/// `IndexOptions.compareTo`: the declaration order, with
/// `DOCS_AND_CUSTOM_FREQS` counted as `DOCS_AND_FREQS` for `subsumes`.
pub(crate) fn index_options_subsumes(this: IndexOptions, other: IndexOptions) -> bool {
    let rank = |o: IndexOptions| match o {
        IndexOptions::None => 0,
        IndexOptions::Docs => 1,
        IndexOptions::DocsAndFreqs | IndexOptions::DocsAndCustomFreqs => 2,
        IndexOptions::DocsAndFreqsAndPositions => 3,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets => 4,
    };
    rank(this) >= rank(other)
}

/// One token of a field's token stream: the attributes `IndexingChain`
/// reads (`TermToBytesRefAttribute`, `OffsetAttribute`,
/// `PositionIncrementAttribute`, `TermFrequencyAttribute`,
/// `PayloadAttribute`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldToken {
    pub term: Vec<u8>,
    pub start_offset: i32,
    pub end_offset: i32,
    pub position_increment: i32,
    pub term_frequency: i32,
    /// `PayloadAttribute.getPayload()`: an empty payload is no payload, as
    /// `FreqProxTermsWriterPerField.writeProx` reads it.
    pub payload: Option<Vec<u8>>,
}

impl FieldToken {
    /// A token at increment 1, frequency 1.
    pub fn new(term: impl Into<Vec<u8>>, start_offset: i32, end_offset: i32) -> Self {
        FieldToken {
            term: term.into(),
            start_offset,
            end_offset,
            position_increment: 1,
            term_frequency: 1,
            payload: None,
        }
    }
}

/// A whole `TokenStream`: its tokens and the two values `end()` leaves in
/// the position-increment and offset attributes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FieldTokens {
    pub tokens: Vec<FieldToken>,
    pub final_position_increment: i32,
    pub final_offset: i32,
    /// Every attribute as `end()` left it -- `FieldInvertState`'s
    /// `getAttributeSource()` after this value. `None` for a stream built
    /// by hand, whose `end()` is taken to be `TokenStream.end()` plus the
    /// two final values ([`crate::similarity::end_attributes`]).
    pub end_attributes: Option<lucene_analysis::AttributeSource>,
}

impl FieldTokens {
    /// [`Self::end_attributes`], or the plain `end()` state of the two final
    /// values.
    pub fn attributes_at_end(&self) -> lucene_analysis::AttributeSource {
        self.end_attributes.clone().unwrap_or_else(|| {
            crate::similarity::end_attributes(self.final_position_increment, self.final_offset)
        })
    }
}

/// `analyzer.tokenStream(field, text)` as `IndexingChain.invertTokenStream`
/// consumes it: `reset()`, every token's term bytes, offsets, increment,
/// `TermFrequencyAttribute` and `PayloadAttribute` -- whatever a
/// `TokenFilter` in the analyzer's chain set -- then `end()`'s attributes and
/// `close()`. An analysis error refuses the document, as Java's exception
/// does.
pub(crate) fn analyze_field(analyzer: &Analyzer, field: &str, text: &str) -> Result<FieldTokens> {
    use lucene_analysis::TokenStream;
    let analysis = |e: lucene_analysis::AnalysisError| illegal(e.to_string());
    let mut ts = analyzer.token_stream(field, text).map_err(analysis)?;
    ts.reset().map_err(analysis)?;
    let mut tokens = Vec::new();
    while ts.increment_token().map_err(analysis)? {
        let a = ts.attributes();
        tokens.push(FieldToken {
            term: a.term_bytes().to_vec(),
            start_offset: a.start_offset(),
            end_offset: a.end_offset(),
            position_increment: a.position_increment(),
            term_frequency: a.term_frequency(),
            payload: a.payload().map(<[u8]>::to_vec),
        });
    }
    ts.end().map_err(analysis)?;
    let end = ts.attributes().clone();
    ts.close().map_err(analysis)?;
    Ok(FieldTokens {
        tokens,
        final_position_increment: end.position_increment(),
        final_offset: end.end_offset(),
        end_attributes: Some(end),
    })
}

impl From<lucene_analysis::AnalyzedTokens> for FieldTokens {
    fn from(stream: lucene_analysis::AnalyzedTokens) -> Self {
        FieldTokens {
            tokens: stream
                .tokens
                .into_iter()
                .map(|t| FieldToken {
                    term: t.term.into_bytes(),
                    start_offset: t.start_offset,
                    end_offset: t.end_offset,
                    position_increment: t.position_increment,
                    term_frequency: 1,
                    payload: None,
                })
                .collect(),
            final_position_increment: stream.final_position_increment,
            final_offset: stream.final_offset,
            end_attributes: None,
        }
    }
}

/// `IndexableField`: what the indexing chain reads from one field instance.
pub trait IndexableField: fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn field_type(&self) -> &FieldType;
    /// `stringValue()`.
    fn string_value(&self) -> Option<Cow<'_, str>> {
        None
    }
    /// `binaryValue()`: the packed point, the doc value, or the binary term.
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        None
    }
    /// `numericValue()`: the doc value of a numeric field.
    fn numeric_value(&self) -> Option<Number> {
        None
    }
    /// `storedValue()`, read when the type is stored.
    fn stored_value(&self) -> Option<StoredValue> {
        None
    }
    /// `invertableType()`.
    fn invertable_type(&self) -> InvertableType {
        InvertableType::TokenStream
    }
    /// `tokenStream(analyzer, reuse)`: `None` for a field that is not
    /// indexed.
    fn token_stream(&self, analyzer: &Analyzer) -> Result<Option<FieldTokens>>;
    /// The KNN vector of a vector field (`KnnFloatVectorField.vectorValue()`
    /// / `KnnByteVectorField.vectorValue()`).
    fn vector_value(&self) -> Option<crate::index_writer::VectorValue> {
        None
    }
}

/// A `Field`'s value (`fieldsData`).
#[derive(Debug, Clone, PartialEq)]
pub enum FieldData {
    String(String),
    /// A `Reader` value: read as text, never stored.
    Reader(String),
    Bytes(Vec<u8>),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    /// A pre-analyzed `TokenStream` value.
    TokenStream(FieldTokens),
}

/// `Field`: a name, a type and a value -- the general field every typed
/// field of the package specialises.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    name: String,
    field_type: FieldType,
    data: FieldData,
}

impl Field {
    /// A field whose `fieldsData` a subclass fills in (`Field(name, type)`).
    pub(crate) fn raw(name: impl Into<String>, field_type: FieldType, data: FieldData) -> Self {
        Field {
            name: name.into(),
            field_type,
            data,
        }
    }

    /// `Field(name, CharSequence value, type)`.
    pub fn from_string(
        name: impl Into<String>,
        value: impl Into<String>,
        field_type: FieldType,
    ) -> Result<Self> {
        if !field_type.stored() && field_type.index_options() == IndexOptions::None {
            return Err(illegal(
                "it doesn't make sense to have a field that is neither indexed nor stored",
            ));
        }
        Ok(Self::raw(name, field_type, FieldData::String(value.into())))
    }

    /// `Field(name, Reader reader, type)`: a text value that cannot be
    /// stored.
    pub fn from_reader(
        name: impl Into<String>,
        text: impl Into<String>,
        field_type: FieldType,
    ) -> Result<Self> {
        if field_type.stored() {
            return Err(illegal("fields with a Reader value cannot be stored"));
        }
        if field_type.index_options() != IndexOptions::None && !field_type.tokenized() {
            return Err(illegal("non-tokenized fields must use String values"));
        }
        Ok(Self::raw(name, field_type, FieldData::Reader(text.into())))
    }

    /// `Field(name, TokenStream tokenStream, type)`.
    pub fn from_token_stream(
        name: impl Into<String>,
        tokens: FieldTokens,
        field_type: FieldType,
    ) -> Result<Self> {
        if field_type.index_options() == IndexOptions::None || !field_type.tokenized() {
            return Err(illegal("TokenStream fields must be indexed and tokenized"));
        }
        if field_type.stored() {
            return Err(illegal("TokenStream fields cannot be stored"));
        }
        Ok(Self::raw(name, field_type, FieldData::TokenStream(tokens)))
    }

    /// `Field(name, BytesRef bytes, type)`.
    pub fn from_bytes(
        name: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
        field_type: FieldType,
    ) -> Result<Self> {
        if index_options_subsumes(
            field_type.index_options(),
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        ) || field_type.store_term_vector_offsets()
        {
            return Err(illegal(
                "It doesn't make sense to index offsets on binary fields",
            ));
        }
        if field_type.index_options() != IndexOptions::None && field_type.tokenized() {
            return Err(illegal("cannot set a BytesRef value on a tokenized field"));
        }
        if field_type.index_options() == IndexOptions::None
            && field_type.point_dimension_count() == 0
            && field_type.doc_values_type() == DocValuesType::None
            && !field_type.stored()
        {
            return Err(illegal(
                "it doesn't make sense to have a field that is neither indexed, nor doc-valued, \
                 nor stored",
            ));
        }
        Ok(Self::raw(name, field_type, FieldData::Bytes(bytes.into())))
    }

    /// A stored-only field (`StoredField`'s types): the value as given.
    pub fn stored(name: impl Into<String>, value: StoredValue) -> Self {
        let mut ft = FieldType::new();
        ft.stored = true;
        let data = match value {
            StoredValue::String(s) => FieldData::String(s),
            StoredValue::Binary(b) => FieldData::Bytes(b),
            StoredValue::Int(v) => FieldData::Int(v),
            StoredValue::Long(v) => FieldData::Long(v),
            StoredValue::Float(v) => FieldData::Float(v),
            StoredValue::Double(v) => FieldData::Double(v),
        };
        Self::raw(name, ft.frozen(), data)
    }

    /// `fieldsData`.
    pub fn data(&self) -> &FieldData {
        &self.data
    }

    /// `setStringValue`.
    pub fn set_string_value(&mut self, value: impl Into<String>) -> Result<()> {
        match &mut self.data {
            FieldData::String(s) => {
                *s = value.into();
                Ok(())
            }
            other => Err(type_change(other, "String")),
        }
    }

    /// `setBytesValue`.
    pub fn set_bytes_value(&mut self, value: impl Into<Vec<u8>>) -> Result<()> {
        match &mut self.data {
            FieldData::Bytes(b) => {
                *b = value.into();
                Ok(())
            }
            other => Err(type_change(other, "BytesRef")),
        }
    }

    /// `setIntValue`.
    pub fn set_int_value(&mut self, value: i32) -> Result<()> {
        match &mut self.data {
            FieldData::Int(v) => {
                *v = value;
                Ok(())
            }
            other => Err(type_change(other, "Integer")),
        }
    }

    /// `setLongValue`.
    pub fn set_long_value(&mut self, value: i64) -> Result<()> {
        match &mut self.data {
            FieldData::Long(v) => {
                *v = value;
                Ok(())
            }
            other => Err(type_change(other, "Long")),
        }
    }

    /// `setFloatValue`.
    pub fn set_float_value(&mut self, value: f32) -> Result<()> {
        match &mut self.data {
            FieldData::Float(v) => {
                *v = value;
                Ok(())
            }
            other => Err(type_change(other, "Float")),
        }
    }

    /// `setDoubleValue`.
    pub fn set_double_value(&mut self, value: f64) -> Result<()> {
        match &mut self.data {
            FieldData::Double(v) => {
                *v = value;
                Ok(())
            }
            other => Err(type_change(other, "Double")),
        }
    }

    /// `setTokenStream`.
    pub fn set_token_stream(&mut self, tokens: FieldTokens) -> Result<()> {
        match &mut self.data {
            FieldData::TokenStream(t) => {
                *t = tokens;
                Ok(())
            }
            other => Err(type_change(other, "TokenStream")),
        }
    }
}

fn data_class(d: &FieldData) -> &'static str {
    match d {
        FieldData::String(_) => "String",
        FieldData::Reader(_) => "Reader",
        FieldData::Bytes(_) => "BytesRef",
        FieldData::Int(_) => "Integer",
        FieldData::Long(_) => "Long",
        FieldData::Float(_) => "Float",
        FieldData::Double(_) => "Double",
        FieldData::TokenStream(_) => "TokenStream",
    }
}

fn type_change(from: &FieldData, to: &str) -> Error {
    illegal(format!(
        "cannot change value type from {} to {to}",
        data_class(from)
    ))
}

impl IndexableField for Field {
    fn name(&self) -> &str {
        &self.name
    }

    fn field_type(&self) -> &FieldType {
        &self.field_type
    }

    /// `stringValue()`: the text, or a number's decimal form.
    fn string_value(&self) -> Option<Cow<'_, str>> {
        match &self.data {
            FieldData::String(s) => Some(Cow::Borrowed(s)),
            FieldData::Int(v) => Some(Cow::Owned(v.to_string())),
            FieldData::Long(v) => Some(Cow::Owned(v.to_string())),
            FieldData::Float(v) => Some(Cow::Owned(java_float_string(*v))),
            FieldData::Double(v) => Some(Cow::Owned(java_double_string(*v))),
            _ => None,
        }
    }

    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        match &self.data {
            FieldData::Bytes(b) => Some(Cow::Borrowed(b)),
            _ => None,
        }
    }

    fn numeric_value(&self) -> Option<Number> {
        match self.data {
            FieldData::Int(v) => Some(Number::Int(v)),
            FieldData::Long(v) => Some(Number::Long(v)),
            FieldData::Float(v) => Some(Number::Float(v)),
            FieldData::Double(v) => Some(Number::Double(v)),
            _ => None,
        }
    }

    /// `Field.storedValue()`: `None` unless the type is stored.
    fn stored_value(&self) -> Option<StoredValue> {
        if !self.field_type.stored() {
            return None;
        }
        match &self.data {
            FieldData::Int(v) => Some(StoredValue::Int(*v)),
            FieldData::Long(v) => Some(StoredValue::Long(*v)),
            FieldData::Float(v) => Some(StoredValue::Float(*v)),
            FieldData::Double(v) => Some(StoredValue::Double(*v)),
            FieldData::Bytes(b) => Some(StoredValue::Binary(b.clone())),
            FieldData::String(s) => Some(StoredValue::String(s.clone())),
            FieldData::Reader(_) | FieldData::TokenStream(_) => None,
        }
    }

    /// `Field.tokenStream`: a single-token stream for an untokenized string
    /// or bytes value, the analyzer's stream for text, the value itself for
    /// a pre-analyzed stream.
    fn token_stream(&self, analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        if self.field_type.index_options() == IndexOptions::None {
            return Ok(None);
        }
        if !self.field_type.tokenized() {
            if let Some(s) = self.string_value() {
                return Ok(Some(string_token_stream(&s)));
            }
            if let Some(b) = self.binary_value() {
                return Ok(Some(FieldTokens {
                    tokens: vec![FieldToken::new(b.into_owned(), 0, 0)],
                    final_position_increment: 0,
                    final_offset: 0,
                    end_attributes: None,
                }));
            }
            return Err(illegal("Non-Tokenized Fields must have a String value"));
        }
        match &self.data {
            FieldData::TokenStream(t) => Ok(Some(t.clone())),
            FieldData::Reader(text) | FieldData::String(text) => {
                analyze_field(analyzer, &self.name, text).map(Some)
            }
            _ => match self.string_value() {
                Some(s) => analyze_field(analyzer, &self.name, &s).map(Some),
                None => Err(illegal(format!(
                    "Field must have either TokenStream, String, Reader or Number value; got {}",
                    self.name
                ))),
            },
        }
    }
}

/// `Field.StringTokenStream`: the whole value as one token, offsets
/// `[0, length)` in UTF-16 units, and `end()`'s final offset the length.
pub(crate) fn string_token_stream(value: &str) -> FieldTokens {
    let len = utf16_len(value);
    FieldTokens {
        tokens: vec![FieldToken::new(value.as_bytes().to_vec(), 0, len)],
        final_position_increment: 0,
        final_offset: len,
        end_attributes: None,
    }
}

/// `String.length()`: UTF-16 code units, saturated to `i32` as a Java
/// string's length is bounded by it.
pub(crate) fn utf16_len(value: &str) -> i32 {
    i32::try_from(value.encode_utf16().count()).unwrap_or(i32::MAX)
}

/// `Float.toString`: enough of it for a stored-value string (the shortest
/// round-tripping decimal, Java's exponent form outside `[1e-3, 1e7)`).
pub(crate) fn java_float_string(v: f32) -> String {
    java_fp_string(f64::from(v), format!("{v:?}"))
}

/// `Double.toString`, as [`java_float_string`].
pub(crate) fn java_double_string(v: f64) -> String {
    java_fp_string(v, format!("{v:?}"))
}

fn java_fp_string(v: f64, debug: String) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let a = v.abs();
    if a == 0.0 || (1e-3..1e7).contains(&a) {
        // Rust's `{:?}` already prints the shortest round-trip with a `.0`.
        return debug;
    }
    // Java: `d.ddddE[-]n`.
    let sci = format!("{:e}", debug.parse::<f64>().unwrap_or(v));
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let mantissa = if mantissa.contains('.') {
        mantissa.to_string()
    } else {
        format!("{mantissa}.0")
    };
    format!("{mantissa}E{exp}")
}

/// `TextField`: indexed, tokenized, positions; stored when asked.
#[derive(Debug, Clone, PartialEq)]
pub struct TextField;

impl TextField {
    /// `TextField.TYPE_NOT_STORED`.
    pub fn type_not_stored() -> FieldType {
        let mut ft = FieldType::new();
        ft.index_options = IndexOptions::DocsAndFreqsAndPositions;
        ft.tokenized = true;
        ft.frozen()
    }

    /// `TextField.TYPE_STORED`.
    pub fn type_stored() -> FieldType {
        let mut ft = Self::type_not_stored();
        ft.stored = true;
        ft.frozen()
    }

    /// `new TextField(name, value, store)`.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(name: impl Into<String>, value: impl Into<String>, store: Store) -> Field {
        let ft = match store {
            Store::Yes => Self::type_stored(),
            Store::No => Self::type_not_stored(),
        };
        Field::raw(name, ft, FieldData::String(value.into()))
    }

    /// `new TextField(name, reader)`.
    pub fn from_reader(name: impl Into<String>, text: impl Into<String>) -> Field {
        Field::raw(
            name,
            Self::type_not_stored(),
            FieldData::Reader(text.into()),
        )
    }

    /// `new TextField(name, stream)`.
    pub fn from_token_stream(name: impl Into<String>, tokens: FieldTokens) -> Field {
        Field::raw(
            name,
            Self::type_not_stored(),
            FieldData::TokenStream(tokens),
        )
    }
}

/// `StringField`: one untokenized term (its bytes: `invertableType()` is
/// `BINARY`), docs only, no norms; stored when asked.
#[derive(Debug, Clone, PartialEq)]
pub struct StringField {
    name: String,
    field_type: FieldType,
    /// `fieldsData` when it is a string.
    string: Option<String>,
    binary: Vec<u8>,
    stored: Option<StoredValue>,
}

impl StringField {
    /// `StringField.TYPE_NOT_STORED`.
    pub fn type_not_stored() -> FieldType {
        let mut ft = FieldType::new();
        ft.omit_norms = true;
        ft.index_options = IndexOptions::Docs;
        ft.tokenized = false;
        ft.frozen()
    }

    /// `StringField.TYPE_STORED`.
    pub fn type_stored() -> FieldType {
        let mut ft = Self::type_not_stored();
        ft.stored = true;
        ft.frozen()
    }

    fn type_of(store: Store) -> FieldType {
        match store {
            Store::Yes => Self::type_stored(),
            Store::No => Self::type_not_stored(),
        }
    }

    /// `new StringField(name, value, store)`: stored as a string.
    pub fn new(name: impl Into<String>, value: impl Into<String>, store: Store) -> Self {
        let value = value.into();
        StringField {
            name: name.into(),
            field_type: Self::type_of(store),
            binary: value.as_bytes().to_vec(),
            stored: (store == Store::Yes).then(|| StoredValue::String(value.clone())),
            string: Some(value),
        }
    }

    /// `new StringField(name, BytesRef value, store)`: stored as bytes.
    pub fn from_bytes(name: impl Into<String>, value: impl Into<Vec<u8>>, store: Store) -> Self {
        let value = value.into();
        StringField {
            name: name.into(),
            field_type: Self::type_of(store),
            stored: (store == Store::Yes).then(|| StoredValue::Binary(value.clone())),
            binary: value,
            string: None,
        }
    }

    /// `setStringValue`.
    pub fn set_string_value(&mut self, value: impl Into<String>) -> Result<()> {
        if self.string.is_none() {
            return Err(illegal("cannot change value type from BytesRef to String"));
        }
        let value = value.into();
        self.binary = value.as_bytes().to_vec();
        if self.stored.is_some() {
            self.stored = Some(StoredValue::String(value.clone()));
        }
        self.string = Some(value);
        Ok(())
    }

    /// `setBytesValue`.
    pub fn set_bytes_value(&mut self, value: impl Into<Vec<u8>>) -> Result<()> {
        if self.string.is_some() {
            return Err(illegal("cannot change value type from String to BytesRef"));
        }
        self.binary = value.into();
        if self.stored.is_some() {
            self.stored = Some(StoredValue::Binary(self.binary.clone()));
        }
        Ok(())
    }
}

impl IndexableField for StringField {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    fn string_value(&self) -> Option<Cow<'_, str>> {
        self.string.as_deref().map(Cow::Borrowed)
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        Some(Cow::Borrowed(&self.binary))
    }
    fn stored_value(&self) -> Option<StoredValue> {
        self.stored.clone()
    }
    fn invertable_type(&self) -> InvertableType {
        InvertableType::Binary
    }
    /// `Field.tokenStream` for the untokenized value (unused: the value is
    /// inverted as one binary term).
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(Some(match &self.string {
            Some(s) => string_token_stream(s),
            None => FieldTokens {
                tokens: vec![FieldToken::new(self.binary.clone(), 0, 0)],
                final_position_increment: 0,
                final_offset: 0,
                end_attributes: None,
            },
        }))
    }
}

/// `Document`: the fields of one document, in the order they were added.
#[derive(Debug, Default)]
pub struct Document {
    fields: Vec<Box<dyn IndexableField>>,
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    /// `add(field)`.
    pub fn add(&mut self, field: impl IndexableField + 'static) {
        self.fields.push(Box::new(field));
    }

    /// `add` for an already-boxed field.
    pub fn add_boxed(&mut self, field: Box<dyn IndexableField>) {
        self.fields.push(field);
    }

    /// `removeField(name)`: the first field with that name.
    pub fn remove_field(&mut self, name: &str) {
        if let Some(i) = self.fields.iter().position(|f| f.name() == name) {
            self.fields.remove(i);
        }
    }

    /// `removeFields(name)`.
    pub fn remove_fields(&mut self, name: &str) {
        self.fields.retain(|f| f.name() != name);
    }

    /// `getBinaryValues(name)`.
    pub fn get_binary_values(&self, name: &str) -> Vec<Vec<u8>> {
        self.fields
            .iter()
            .filter(|f| f.name() == name)
            .filter_map(|f| f.binary_value().map(Cow::into_owned))
            .collect()
    }

    /// `getBinaryValue(name)`.
    pub fn get_binary_value(&self, name: &str) -> Option<Vec<u8>> {
        self.fields
            .iter()
            .filter(|f| f.name() == name)
            .find_map(|f| f.binary_value().map(Cow::into_owned))
    }

    /// `getField(name)`.
    pub fn get_field(&self, name: &str) -> Option<&dyn IndexableField> {
        self.fields
            .iter()
            .find(|f| f.name() == name)
            .map(|f| f.as_ref())
    }

    /// `getFields(name)`.
    pub fn get_fields_named(&self, name: &str) -> Vec<&dyn IndexableField> {
        self.fields
            .iter()
            .filter(|f| f.name() == name)
            .map(|f| f.as_ref())
            .collect()
    }

    /// `getFields()`.
    pub fn fields(&self) -> &[Box<dyn IndexableField>] {
        &self.fields
    }

    /// `getValues(name)`.
    pub fn get_values(&self, name: &str) -> Vec<String> {
        self.fields
            .iter()
            .filter(|f| f.name() == name)
            .filter_map(|f| f.string_value().map(Cow::into_owned))
            .collect()
    }

    /// `get(name)`.
    pub fn get(&self, name: &str) -> Option<String> {
        self.fields
            .iter()
            .filter(|f| f.name() == name)
            .find_map(|f| f.string_value().map(Cow::into_owned))
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.fields.clear();
    }
}

#[cfg(test)]
mod tests;
