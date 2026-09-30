//! `KnnFloatVectorField` and `KnnByteVectorField` as document-API fields:
//! one vector per field and document, indexed by the writer's flat and HNSW
//! vector writers (the same path as
//! [`crate::index_writer::IndexWriter::add_document_with_vectors`]).

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
        if let Some(bad) = vector.iter().find(|v| !v.is_finite()) {
            return Err(illegal(format!(
                "non-finite value at vector[{}]={bad}",
                vector.iter().position(|v| !v.is_finite()).unwrap_or(0)
            )));
        }
        Ok(KnnFloatVectorField {
            name: name.into(),
            field_type: vector_type(vector.len(), VectorEncoding::Float32, similarity)?,
            vector,
        })
    }

    /// Over a given field type (`KnnFloatVectorField(name, vector, fieldType)`).
    pub fn with_type(name: impl Into<String>, vector: Vec<f32>, field_type: FieldType) -> Self {
        KnnFloatVectorField {
            name: name.into(),
            field_type,
            vector,
        }
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

    /// Over a given field type.
    pub fn with_type(name: impl Into<String>, vector: Vec<u8>, field_type: FieldType) -> Self {
        KnnByteVectorField {
            name: name.into(),
            field_type,
            vector,
        }
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
        let t = KnnByteVectorField::with_type("b", vec![1], b.field_type().clone());
        assert_eq!(t.vector(), &[1]);
        let t = KnnFloatVectorField::with_type("v", vec![1.0], f.field_type().clone());
        assert_eq!(t.vector(), &[1.0]);
    }
}
