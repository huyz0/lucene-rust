//! `LateInteractionField`: a document's multi-vector (one vector per token,
//! as late-interaction models such as ColBERT produce), stored as one
//! `BINARY` doc value.
//!
//! What reaches disk: a little-endian `int` token-vector dimension, then
//! every vector's `float`s, little-endian, in order.

use std::borrow::Cow;

use lucene_analysis::Analyzer;

use super::doc_values::BinaryDocValuesField;
use super::{illegal, FieldTokens, FieldType, IndexableField, Result};

/// `LateInteractionField`.
#[derive(Debug, Clone, PartialEq)]
pub struct LateInteractionField(BinaryDocValuesField);

impl LateInteractionField {
    /// `new LateInteractionField(name, value)`.
    pub fn new(name: impl Into<String>, value: &[Vec<f32>]) -> Result<Self> {
        Ok(LateInteractionField(BinaryDocValuesField::new(
            name,
            Self::encode(value)?,
        )))
    }

    /// `setValue(value)`.
    pub fn set_value(&mut self, value: &[Vec<f32>]) -> Result<()> {
        self.0.set_value(Self::encode(value)?);
        Ok(())
    }

    /// `getValue()`.
    pub fn value(&self) -> Result<Vec<Vec<f32>>> {
        Self::decode(self.0.value())
    }

    /// `encode(value)`.
    pub fn encode(value: &[Vec<f32>]) -> Result<Vec<u8>> {
        let Some(first) = value.first() else {
            return Err(illegal("Value should not be null or empty"));
        };
        if first.is_empty() {
            return Err(illegal(
                "Composing token vectors should not be null or empty",
            ));
        }
        let dim = first.len();
        let dim_i32 = i32::try_from(dim).map_err(|_| illegal("token vector too long"))?;
        let mut out = Vec::with_capacity(
            value
                .len()
                .saturating_mul(dim)
                .saturating_mul(4)
                .saturating_add(4),
        );
        out.extend_from_slice(&dim_i32.to_le_bytes());
        for (i, v) in value.iter().enumerate() {
            if v.len() != dim {
                return Err(illegal(format!(
                    "Composing token vectors should have the same dimension. Mismatching \
                     dimensions detected between token[0] and token[{i}], {dim} != {}",
                    v.len()
                )));
            }
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// `decode(payload)`.
    pub fn decode(payload: &[u8]) -> Result<Vec<Vec<f32>>> {
        let Some(head) = payload.get(..4) else {
            return Err(illegal("payload shorter than its dimension"));
        };
        let dim = i32::from_le_bytes(head.try_into().expect("four bytes"));
        let body = &payload[4..];
        let stride = usize::try_from(dim)
            .ok()
            .and_then(|d| d.checked_mul(4))
            .filter(|&s| s > 0);
        let Some(stride) = stride.filter(|&s| body.len().checked_rem(s) == Some(0)) else {
            return Err(illegal(format!(
                "Provided payload does not appear to have been encoded via \
                 LateInteractionField.encode. Payload length should be equal to 4 + numVectors \
                 * tokenVectorDimension, got {} != 4 + ? * {dim}",
                payload.len()
            )));
        };
        Ok(body
            .chunks_exact(stride)
            .map(|v| {
                v.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                    .collect()
            })
            .collect())
    }
}

impl IndexableField for LateInteractionField {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn field_type(&self) -> &FieldType {
        self.0.field_type()
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        self.0.binary_value()
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let v = vec![vec![1.0, -2.0], vec![0.5, 4.0], vec![3.0, 3.5]];
        let mut f = LateInteractionField::new("li", &v).unwrap();
        assert_eq!(f.value().unwrap(), v);
        let bytes = f.binary_value().unwrap().into_owned();
        assert_eq!(&bytes[..4], &2i32.to_le_bytes());
        assert_eq!(bytes.len(), 4 + 6 * 4);
        f.set_value(&[vec![9.0]]).unwrap();
        assert_eq!(f.value().unwrap(), vec![vec![9.0]]);
        assert_eq!(f.name(), "li");
        assert!(f.token_stream(&Analyzer::keyword()).unwrap().is_none());
        assert!(f.field_type().is_frozen());
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(LateInteractionField::encode(&[]).is_err());
        assert!(LateInteractionField::encode(&[vec![]]).is_err());
        assert!(LateInteractionField::encode(&[vec![1.0], vec![1.0, 2.0]]).is_err());
        assert!(LateInteractionField::decode(&[1]).is_err());
        assert!(LateInteractionField::decode(&[0, 0, 0, 0]).is_err());
        let mut odd = 2i32.to_le_bytes().to_vec();
        odd.extend_from_slice(&[0; 4]);
        assert!(LateInteractionField::decode(&odd).is_err());
        assert_eq!(
            LateInteractionField::decode(&1i32.to_le_bytes()).unwrap(),
            Vec::<Vec<f32>>::new()
        );
    }
}
