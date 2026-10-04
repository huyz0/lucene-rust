//! `org.apache.lucene.queries.function.valuesource`: every value source of
//! `lucene-queries` -- constants ([`constants`]), doc-values fields
//! ([`fields`]), arithmetic and boolean functions ([`functions`]), term
//! and index statistics ([`terms`]), a query's scores ([`QueryValueSource`])
//! and vectors ([`vectors`]).
//!
//! Java's abstract helper bases with a single abstract hook
//! (`SimpleFloatFunction.func`, `SimpleBoolFunction.func`,
//! `MultiBoolFunction.func`, `ComparisonBoolFunction.compare`,
//! `MultiFunction` with its `name()`) are concrete structs taking the hook as
//! a function, so a caller writes the subclass Java would.

pub mod constants;
pub mod fields;
pub mod functions;
mod query;
pub mod terms;
pub mod vectors;

pub use constants::{
    ConstKnnByteVectorValueSource, ConstKnnFloatValueSource, ConstNumberSource, ConstValueSource,
    DoubleConstValueSource, LiteralValueSource,
};
pub use fields::{
    BytesRefFieldSource, DoubleFieldSource, EnumFieldSource, FieldCacheSource, FloatFieldSource,
    IntFieldSource, JoinDocFreqValueSource, LongFieldSource, MultiValuedDoubleFieldSource,
    MultiValuedFloatFieldSource, MultiValuedIntFieldSource, MultiValuedLongFieldSource,
    NumericSelector, SetSelector, SortedSetFieldSource,
};
pub use functions::{
    BoolFunction, ComparisonBoolFunction, DefFunction, DivFloatFunction, DualFloatFunction,
    IfFunction, LinearFloatFunction, MaxFloatFunction, MinFloatFunction, MultiBoolFunction,
    MultiFloatFunction, MultiFunction, PowFloatFunction, ProductFloatFunction,
    RangeMapFloatFunction, ReciprocalFloatFunction, ScaleFloatFunction, SimpleBoolFunction,
    SimpleFloatFunction, SingleFunction, SumFloatFunction,
};
pub use query::QueryValueSource;
pub use terms::{
    DocFreqValueSource, IDFValueSource, MaxDocValueSource, NormValueSource, NumDocsValueSource,
    SumTotalTermFreqValueSource, TFValueSource, TermFreqValueSource, TotalTermFreqValueSource,
};
pub use vectors::{
    ByteKnnVectorFieldSource, ByteVectorSimilarityFunction, FloatKnnVectorFieldSource,
    FloatVectorSimilarityFunction, MultiValueSource, VectorFieldFunction, VectorSimilarityFunction,
    VectorValueSource,
};
