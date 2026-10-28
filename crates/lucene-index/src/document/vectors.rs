//! `KnnFloatVectorField` and `KnnByteVectorField` as document-API fields:
//! one vector per field and document, indexed by the writer's flat and HNSW
//! vector writers (the same path as
//! [`crate::index_writer::IndexWriter::add_document_with_vectors`]).
//!
//! [`IndexableField::vector_value`] hands the writer an owned copy of the
//! vector: the field is borrowed for the call while the writer buffers the
//! vector until flush, so one copy per document is needed, the copy Java's
//! `FlatFieldVectorsWriter.addValue` makes too (`copyValue`).

use lucene_analysis::Analyzer;

use super::{
    illegal, FieldTokens, FieldType, IndexableField, Result, VectorEncoding,
    VectorSimilarityFunction,
};
use crate::index_writer::VectorValue;

fn vector_type(
    dimension: usize,
    encoding: VectorEncoding,
    similarity: VectorSimilarityFunction,
) -> Result<FieldType> {
    let dim = i32::try_from(dimension).map_err(|_| illegal("vector too long"))?;
    if dim == 0 {
        return Err(illegal("cannot index an empty vector"));
    }
    let mut ft = FieldType::new();
    ft.set_vector_attributes(dim, encoding, similarity)?;
    Ok(ft.frozen())
}

/// `VectorUtil.checkFinite(vector)`: `non-finite value at vector[i]=v`, `v`
/// as Java's `Float.toString` prints a non-finite float.
fn check_finite(vector: &[f32]) -> Result<()> {
    let Some((i, v)) = vector.iter().enumerate().find(|(_, v)| !v.is_finite()) else {
        return Ok(());
    };
    let v = if v.is_nan() {
        "NaN"
    } else if *v > 0.0 {
        "Infinity"
    } else {
        "-Infinity"
    };
    Err(illegal(format!("non-finite value at vector[{i}]={v}")))
}

/// The `(name, vector, fieldType)` constructors' encoding check.
fn check_encoding(
    name: &str,
    kind: &str,
    field_type: &FieldType,
    expected: VectorEncoding,
) -> Result<()> {
    let encoding = field_type.vector_encoding();
    if encoding == expected {
        return Ok(());
    }
    let encoding = match encoding {
        VectorEncoding::Byte => "BYTE",
        VectorEncoding::Float32 => "FLOAT32",
    };
    Err(illegal(format!(
        "Attempt to create a vector for field {name} using {kind} but the field encoding is \
         {encoding}"
    )))
}

/// `KnnFloatVectorField`.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnFloatVectorField {
    name: String,
    field_type: FieldType,
    vector: Vec<f32>,
}

impl KnnFloatVectorField {
    /// `new KnnFloatVectorField(name, vector, similarityFunction)`: every
    /// component finite.
    pub fn new(
        name: impl Into<String>,
        vector: Vec<f32>,
        similarity: VectorSimilarityFunction,
    ) -> Result<Self> {
        check_finite(&vector)?;
        Ok(KnnFloatVectorField {
            name: name.into(),
            field_type: vector_type(vector.len(), VectorEncoding::Float32, similarity)?,
            vector,
        })
    }

    /// Over a given field type (`KnnFloatVectorField(name, vector, fieldType)`).
    ///
    /// # Errors
    /// A field type whose vector encoding is not `FLOAT32`, or a non-finite
    /// component (`VectorUtil.checkFinite`), as Java's constructor refuses
    /// them. The dimension is the writer's check, as in Java.
    pub fn with_type(
        name: impl Into<String>,
        vector: Vec<f32>,
        field_type: FieldType,
    ) -> Result<Self> {
        let name = name.into();
        check_encoding(&name, "float[]", &field_type, VectorEncoding::Float32)?;
        check_finite(&vector)?;
        Ok(KnnFloatVectorField {
            name,
            field_type,
            vector,
        })
    }

    pub fn vector(&self) -> &[f32] {
        &self.vector
    }
}

impl IndexableField for KnnFloatVectorField {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    fn vector_value(&self) -> Option<VectorValue> {
        Some(VectorValue::Float32(self.vector.clone()))
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

/// `KnnByteVectorField`: bytes held unsigned, as `.vec` stores them.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnByteVectorField {
    name: String,
    field_type: FieldType,
    vector: Vec<u8>,
}

impl KnnByteVectorField {
    /// `new KnnByteVectorField(name, vector, similarityFunction)`.
    pub fn new(
        name: impl Into<String>,
        vector: Vec<u8>,
        similarity: VectorSimilarityFunction,
    ) -> Result<Self> {
        Ok(KnnByteVectorField {
            name: name.into(),
            field_type: vector_type(vector.len(), VectorEncoding::Byte, similarity)?,
            vector,
        })
    }

    /// Over a given field type (`KnnByteVectorField(name, vector, fieldType)`).
    ///
    /// # Errors
    /// A field type whose vector encoding is not `BYTE`, as Java's
    /// constructor refuses it.
    pub fn with_type(
        name: impl Into<String>,
        vector: Vec<u8>,
        field_type: FieldType,
    ) -> Result<Self> {
        let name = name.into();
        check_encoding(&name, "byte[]", &field_type, VectorEncoding::Byte)?;
        Ok(KnnByteVectorField {
            name,
            field_type,
            vector,
        })
    }

    pub fn vector(&self) -> &[u8] {
        &self.vector
    }
}

impl IndexableField for KnnByteVectorField {
    fn name(&self) -> &str {
        &self.name
    }
    fn field_type(&self) -> &FieldType {
        &self.field_type
    }
    fn vector_value(&self) -> Option<VectorValue> {
        Some(VectorValue::Byte(self.vector.clone()))
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_fields_carry_their_shape() {
        let f = KnnFloatVectorField::new("v", vec![1.0, 2.0], VectorSimilarityFunction::Cosine)
            .unwrap();
        assert_eq!(f.field_type().vector_dimension(), 2);
        assert_eq!(f.vector(), &[1.0, 2.0]);
        assert_eq!(f.vector_value(), Some(VectorValue::Float32(vec![1.0, 2.0])));
        assert_eq!(f.name(), "v");
        assert!(f.token_stream(&Analyzer::keyword()).unwrap().is_none());
        assert!(KnnFloatVectorField::new("v", vec![], VectorSimilarityFunction::Cosine).is_err());
        assert!(
            KnnFloatVectorField::new("v", vec![f32::NAN], VectorSimilarityFunction::Cosine)
                .is_err()
        );
        let b = KnnByteVectorField::new("b", vec![1, 255], VectorSimilarityFunction::DotProduct)
            .unwrap();
        assert_eq!(b.field_type().vector_encoding(), VectorEncoding::Byte);
        assert_eq!(b.vector(), &[1, 255]);
        assert_eq!(b.vector_value(), Some(VectorValue::Byte(vec![1, 255])));
        assert_eq!(b.name(), "b");
        assert!(b.token_stream(&Analyzer::keyword()).unwrap().is_none());
        let t = KnnByteVectorField::with_type("b", vec![1], b.field_type().clone()).unwrap();
        assert_eq!(t.vector(), &[1]);
        let t = KnnFloatVectorField::with_type("v", vec![1.0], f.field_type().clone()).unwrap();
        assert_eq!(t.vector(), &[1.0]);
    }

    /// The `(name, vector, fieldType)` constructors refuse what Java's do.
    #[test]
    fn typed_constructors_check_encoding_and_finiteness() {
        let float_type = KnnFloatVectorField::new("v", vec![1.0], VectorSimilarityFunction::Cosine)
            .unwrap()
            .field_type()
            .clone();
        let byte_type = KnnByteVectorField::new("b", vec![1], VectorSimilarityFunction::Cosine)
            .unwrap()
            .field_type()
            .clone();
        let e = KnnByteVectorField::with_type("b", vec![1], float_type.clone()).unwrap_err();
        assert!(
            e.to_string().contains(
                "Attempt to create a vector for field b using byte[] but the field encoding is \
                 FLOAT32"
            ),
            "{e}"
        );
        let e = KnnFloatVectorField::with_type("v", vec![1.0], byte_type).unwrap_err();
        assert!(
            e.to_string().contains(
                "Attempt to create a vector for field v using float[] but the field encoding is \
                 BYTE"
            ),
            "{e}"
        );
        let e = KnnFloatVectorField::with_type("v", vec![1.0, f32::INFINITY], float_type.clone())
            .unwrap_err();
        assert!(
            e.to_string()
                .contains("non-finite value at vector[1]=Infinity"),
            "{e}"
        );
        let e = KnnFloatVectorField::with_type("v", vec![f32::NAN], float_type).unwrap_err();
        assert!(
            e.to_string().contains("non-finite value at vector[0]=NaN"),
            "{e}"
        );
    }
}
