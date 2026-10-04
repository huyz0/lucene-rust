//! The vector sources: `VectorFieldFunction` with
//! `FloatKnnVectorFieldSource`/`ByteKnnVectorFieldSource`,
//! `VectorSimilarityFunction` with `FloatVectorSimilarityFunction`/
//! `ByteVectorSimilarityFunction`, and `MultiValueSource` with
//! `VectorValueSource` (several sources' values as one vector).

use std::sync::Arc;

use lucene_codecs::field_infos::{VectorEncoding, VectorSimilarityFunction as Similarity};

use crate::function::{
    java_float, out_of_order, BoxValues, FunctionContext, FunctionValues, ValueLeaf, ValueSource,
};
use crate::reader::{ByteVectorValues, FloatVectorValues, LeafReader};
use crate::{Error, Result};

/// `VectorEncoding.toString()`.
fn encoding_name(e: VectorEncoding) -> &'static str {
    match e {
        VectorEncoding::Byte => "BYTE",
        VectorEncoding::Float32 => "FLOAT32",
    }
}

/// `VectorSimilarityFunction.name()`.
fn similarity_name(f: Similarity) -> &'static str {
    match f {
        Similarity::Euclidean => "EUCLIDEAN",
        Similarity::DotProduct => "DOT_PRODUCT",
        Similarity::Cosine => "COSINE",
        Similarity::MaximumInnerProduct => "MAXIMUM_INNER_PRODUCT",
    }
}

/// `VectorFieldFunction.checkField(reader, field, expectedEncoding)`: a field
/// that exists but holds no vectors of `expected`'s encoding.
fn check_field(
    reader: &crate::directory_reader::SegmentReader,
    field: &str,
    expected: VectorEncoding,
) -> Result<()> {
    let Some(fi) = LeafReader::field_infos(reader).field_by_name(field) else {
        return Ok(());
    };
    let actual = (fi.vector_dimension != 0).then_some(fi.vector_encoding);
    if actual != Some(expected) {
        return Err(Error::IllegalState(format!(
            "Unexpected vector encoding ({}) for field {field}(expected={})",
            actual.map_or("null", encoding_name),
            encoding_name(expected)
        )));
    }
    Ok(())
}

/// A field's vectors by document: the ordinals walked forward as the
/// documents are (`KnnVectorValues.DocIndexIterator`).
struct VectorCursor {
    size: i32,
    ord: i32,
    doc: i32,
}

impl VectorCursor {
    fn new(size: i32) -> Self {
        Self {
            size,
            ord: -1,
            doc: -1,
        }
    }

    /// `advance(target)` over `ord_to_doc`.
    fn advance(&mut self, target: i32, ord_to_doc: impl Fn(i32) -> Result<i32>) -> Result<i32> {
        loop {
            self.ord += 1;
            if self.ord >= self.size {
                self.doc = crate::reader::NO_MORE_DOCS;
                return Ok(self.doc);
            }
            self.doc = ord_to_doc(self.ord)?;
            if self.doc >= target {
                return Ok(self.doc);
            }
        }
    }
}

/// `VectorFieldFunction`: a vector field's values, `exists(doc)` walking
/// its documents forward (`docs were sent out-of-order` backwards), and
/// `toString(doc)` its source's description followed by `strVal` (which
/// a vector field's values do not have).
pub struct VectorFieldFunction<'a> {
    vectors: VectorsOf<'a>,
    kind: Kind,
    cursor: VectorCursor,
    last_doc: i32,
    description: String,
}

enum VectorsOf<'a> {
    Empty,
    Float(Box<dyn FloatVectorValues + 'a>),
    Byte(Box<dyn ByteVectorValues + 'a>),
}

impl VectorFieldFunction<'_> {
    fn ord_to_doc(&self, ord: i32) -> Result<i32> {
        match &self.vectors {
            VectorsOf::Float(v) => v.ord_to_doc(ord),
            VectorsOf::Byte(v) => v.ord_to_doc(ord),
            VectorsOf::Empty => Ok(crate::reader::NO_MORE_DOCS),
        }
    }
}

impl FunctionValues for VectorFieldFunction<'_> {
    fn exists(&mut self, doc: i32) -> Result<bool> {
        if doc < self.last_doc {
            return Err(out_of_order(self.last_doc, doc));
        }
        self.last_doc = doc;
        let mut cur = self.cursor.doc;
        if doc > cur {
            let mut cursor = std::mem::replace(&mut self.cursor, VectorCursor::new(0));
            let r = cursor.advance(doc, |o| self.ord_to_doc(o));
            self.cursor = cursor;
            cur = r?;
        }
        Ok(doc == cur)
    }
    fn float_vector_val(&mut self, doc: i32) -> Result<Option<Vec<f32>>> {
        if !matches!(self.kind, Kind::Float) {
            return Err(crate::function::unsupported());
        }
        if matches!(self.vectors, VectorsOf::Empty) || !self.exists(doc)? {
            return Ok(None);
        }
        match &self.vectors {
            VectorsOf::Float(v) => Ok(Some(v.vector_value(self.cursor.ord)?)),
            _ => Ok(None),
        }
    }
    fn byte_vector_val(&mut self, doc: i32) -> Result<Option<Vec<u8>>> {
        if !matches!(self.kind, Kind::Byte) {
            return Err(crate::function::unsupported());
        }
        if matches!(self.vectors, VectorsOf::Empty) || !self.exists(doc)? {
            return Ok(None);
        }
        match &self.vectors {
            VectorsOf::Byte(v) => Ok(Some(v.vector_value(self.cursor.ord)?)),
            _ => Ok(None),
        }
    }
    /// `description + strVal(doc)`: a vector field's values have no
    /// `strVal` (`FunctionValues`' own throws), so neither has this.
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?.unwrap_or_default();
        Ok(format!("{}{s}", self.description))
    }
}

/// Which vector getter a field's values answer (the other kind's is
/// unsupported).
#[derive(Clone, Copy)]
enum Kind {
    Float,
    Byte,
}

fn vector_field<'a>(
    leaf: &ValueLeaf<'a>,
    field: &str,
    kind: Kind,
    description: String,
) -> Result<BoxValues<'a>> {
    let reader = leaf.reader()?;
    let vectors = match kind {
        Kind::Float => LeafReader::float_vector_values(reader, field)?.map(VectorsOf::Float),
        Kind::Byte => LeafReader::byte_vector_values(reader, field)?.map(VectorsOf::Byte),
    };
    let (vectors, size) = match vectors {
        Some(VectorsOf::Float(v)) => {
            let n = v.size();
            (VectorsOf::Float(v), n)
        }
        Some(VectorsOf::Byte(v)) => {
            let n = v.size();
            (VectorsOf::Byte(v), n)
        }
        _ => {
            let expected = match kind {
                Kind::Float => VectorEncoding::Float32,
                Kind::Byte => VectorEncoding::Byte,
            };
            check_field(reader, field, expected)?;
            (VectorsOf::Empty, 0)
        }
    };
    Ok(Box::new(VectorFieldFunction {
        vectors,
        kind,
        cursor: VectorCursor::new(size),
        last_doc: 0,
        description,
    }))
}

/// `FloatKnnVectorFieldSource`: a `float` vector field's vectors.
#[derive(Debug, Clone)]
pub struct FloatKnnVectorFieldSource {
    field_name: String,
}

impl FloatKnnVectorFieldSource {
    pub fn new(field_name: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
        }
    }
}

impl ValueSource for FloatKnnVectorFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        vector_field(leaf, &self.field_name, Kind::Float, self.description())
    }
    fn description(&self) -> String {
        format!("FloatKnnVectorFieldSource({})", self.field_name)
    }
}

/// `ByteKnnVectorFieldSource`: a `byte` vector field's vectors.
#[derive(Debug, Clone)]
pub struct ByteKnnVectorFieldSource {
    field_name: String,
}

impl ByteKnnVectorFieldSource {
    pub fn new(field_name: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
        }
    }
}

impl ValueSource for ByteKnnVectorFieldSource {
    fn get_values<'a>(
        &self,
        _fcx: &FunctionContext,
        leaf: &ValueLeaf<'a>,
    ) -> Result<BoxValues<'a>> {
        vector_field(leaf, &self.field_name, Kind::Byte, self.description())
    }
    fn description(&self) -> String {
        format!("ByteKnnVectorFieldSource({})", self.field_name)
    }
}

// ---------------------------------------------------------------------------
// VectorSimilarityFunction
// ---------------------------------------------------------------------------

/// `VectorSimilarityFunction`: the similarity of two sources' vectors (`0`
/// where either has none); a document has a value when both sources do.
#[derive(Clone)]
pub struct VectorSimilarityFunction {
    similarity: Similarity,
    vector1: Arc<dyn ValueSource>,
    vector2: Arc<dyn ValueSource>,
    kind: Kind,
}

/// `FloatVectorSimilarityFunction`: over `floatVectorVal`s.
pub struct FloatVectorSimilarityFunction;

impl FloatVectorSimilarityFunction {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        similarity: Similarity,
        vector1: Arc<dyn ValueSource>,
        vector2: Arc<dyn ValueSource>,
    ) -> VectorSimilarityFunction {
        VectorSimilarityFunction {
            similarity,
            vector1,
            vector2,
            kind: Kind::Float,
        }
    }
}

/// `ByteVectorSimilarityFunction`: over `byteVectorVal`s.
pub struct ByteVectorSimilarityFunction;

impl ByteVectorSimilarityFunction {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        similarity: Similarity,
        vector1: Arc<dyn ValueSource>,
        vector2: Arc<dyn ValueSource>,
    ) -> VectorSimilarityFunction {
        VectorSimilarityFunction {
            similarity,
            vector1,
            vector2,
            kind: Kind::Byte,
        }
    }
}

/// `VectorUtil`'s dimension check: `vector dimensions differ: a!=b`.
fn same_dimensions(a: usize, b: usize) -> Result<()> {
    if a == b {
        Ok(())
    } else {
        Err(Error::IllegalArgument(format!(
            "vector dimensions differ: {a}!={b}"
        )))
    }
}

struct VectorSimilarityValues<'a> {
    v1: BoxValues<'a>,
    v2: BoxValues<'a>,
    similarity: Similarity,
    kind: Kind,
    description: String,
}

impl VectorSimilarityValues<'_> {
    /// `func(doc, f1, f2)`. Vectors of different lengths are `VectorUtil`'s
    /// `IllegalArgumentException` (`func`'s own `assert` is off in
    /// production), never a comparison of the common prefix.
    fn func(&mut self, doc: i32) -> Result<f32> {
        Ok(match self.kind {
            Kind::Float => {
                let a = self.v1.float_vector_val(doc)?;
                let b = self.v2.float_vector_val(doc)?;
                match (a, b) {
                    (Some(a), Some(b)) => {
                        same_dimensions(a.len(), b.len())?;
                        self.similarity.score(&a, &b)
                    }
                    _ => 0.0,
                }
            }
            Kind::Byte => {
                let a = self.v1.byte_vector_val(doc)?;
                let b = self.v2.byte_vector_val(doc)?;
                match (a, b) {
                    (Some(a), Some(b)) => {
                        same_dimensions(a.len(), b.len())?;
                        self.similarity.score_bytes(&a, &b)
                    }
                    _ => 0.0,
                }
            }
        })
    }
}

impl FunctionValues for VectorSimilarityValues<'_> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        self.func(doc)
    }
    fn double_val(&mut self, doc: i32) -> Result<f64> {
        Ok(f64::from(self.func(doc)?))
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(Some(java_float(self.func(doc)?)))
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        Ok(self.v1.exists(doc)? && self.v2.exists(doc)?)
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?.unwrap_or_default();
        Ok(format!("{} = {s}", self.description))
    }
}

impl ValueSource for VectorSimilarityFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(VectorSimilarityValues {
            v1: self.vector1.get_values(fcx, leaf)?,
            v2: self.vector2.get_values(fcx, leaf)?,
            similarity: self.similarity,
            kind: self.kind,
            description: self.description(),
        }))
    }
    fn description(&self) -> String {
        format!(
            "{}({}, {})",
            similarity_name(self.similarity),
            self.vector1.description(),
            self.vector2.description()
        )
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![self.vector1.as_ref(), self.vector2.as_ref()]
    }
}

// ---------------------------------------------------------------------------
// MultiValueSource, VectorValueSource
// ---------------------------------------------------------------------------

/// `MultiValueSource`: a source of several values per document.
pub trait MultiValueSource: ValueSource {
    /// `dimension()`.
    fn dimension(&self) -> usize;
}

/// `VectorValueSource`: several sources' values as one vector, read by the
/// multi-valued getters (`floatVal(doc, vals)` and the rest).
#[derive(Clone)]
pub struct VectorValueSource {
    sources: Vec<Arc<dyn ValueSource>>,
}

impl VectorValueSource {
    pub fn new(sources: Vec<Arc<dyn ValueSource>>) -> Self {
        Self { sources }
    }

    /// `getSources()`.
    pub fn get_sources(&self) -> &[Arc<dyn ValueSource>] {
        &self.sources
    }

    /// `name()`.
    pub fn name(&self) -> &'static str {
        "vector"
    }
}

struct VectorValues<'a> {
    values: Vec<BoxValues<'a>>,
}

macro_rules! fill_each {
    ($($m:ident => $single:ident : $ty:ty),*) => {
        $(
            fn $m(&mut self, doc: i32, vals: &mut [$ty]) -> Result<()> {
                for (v, slot) in self.values.iter_mut().zip(vals.iter_mut()) {
                    *slot = v.$single(doc)?;
                }
                Ok(())
            }
        )*
    };
}

impl FunctionValues for VectorValues<'_> {
    fill_each!(
        byte_vals => byte_val: i8,
        short_vals => short_val: i16,
        float_vals => float_val: f32,
        int_vals => int_val: i32,
        long_vals => long_val: i64,
        double_vals => double_val: f64,
        str_vals => str_val: Option<String>
    );
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let mut parts = Vec::with_capacity(self.values.len());
        for v in &mut self.values {
            parts.push(v.to_string_doc(doc)?);
        }
        Ok(format!("vector({})", parts.join(",")))
    }
}

impl ValueSource for VectorValueSource {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(VectorValues {
            values: self
                .sources
                .iter()
                .map(|s| s.get_values(fcx, leaf))
                .collect::<Result<_>>()?,
        }))
    }
    fn description(&self) -> String {
        let parts: Vec<String> = self.sources.iter().map(|s| s.description()).collect();
        format!("{}({})", self.name(), parts.join(","))
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        self.sources.iter().map(|s| s.as_ref()).collect()
    }
}

impl MultiValueSource for VectorValueSource {
    fn dimension(&self) -> usize {
        self.sources.len()
    }
}
