//! `org.apache.lucene.sandbox.document`: the sandbox's point fields and
//! their queries.
//!
//! Each field is an [`IndexableField`](lucene_index::document::IndexableField) the index writer indexes as a point,
//! the way Java's `Field` subclasses are: `name`, the frozen point
//! [`FieldType`] and the packed value (`binaryValue()`). The query
//! factories keep Java's class/method names as module/function names, as
//! [`crate::document`]'s do.

mod big_integer_point;
mod float_point_nearest_neighbor;
mod half_float_point;
mod lat_lon_bounding_box;
mod multi_range_builders;

use lucene_index::document::FieldType;

use crate::{Error, Result};

pub use big_integer_point::BigIntegerPoint;
pub use float_point_nearest_neighbor::{FloatPointNearestNeighbor, NearestNeighborHits};
pub use half_float_point::HalfFloatPoint;
pub use lat_lon_bounding_box::LatLonBoundingBox;
pub use multi_range_builders::{
    DoublePointMultiRangeBuilder, FloatPointMultiRangeBuilder, IntPointMultiRangeBuilder,
    LongPointMultiRangeBuilder,
};

fn illegal(message: impl Into<String>) -> Error {
    Error::DocumentQuery(message.into())
}

/// `getType(numDims)` of a point field: `numDims` dimensions of `bytes`
/// bytes, frozen.
fn point_type(num_dims: usize, bytes: i32) -> Result<FieldType> {
    let dims = i32::try_from(num_dims).map_err(|_| illegal("too many dimensions"))?;
    let mut ft = FieldType::new();
    ft.set_dimensions(dims, bytes)
        .map_err(|e| illegal(e.to_string()))?;
    Ok(ft.frozen())
}

/// The part every sandbox point field shares: Java's `Field` with a
/// `BytesRef` `fieldsData`.
#[derive(Debug, Clone, PartialEq)]
struct PackedField {
    name: String,
    field_type: FieldType,
    packed: Vec<u8>,
}

impl PackedField {
    fn dims(&self) -> usize {
        usize::try_from(self.field_type.point_dimension_count()).unwrap_or(0)
    }

    /// The `setXxxValues(...)` dimension check and message.
    fn check_dims(&self, incoming: usize) -> Result<()> {
        if incoming != self.dims() {
            return Err(illegal(format!(
                "this field (name={}) uses {} dimensions; cannot change to (incoming) {incoming} \
                 dimensions",
                self.name,
                self.dims()
            )));
        }
        Ok(())
    }

    /// The `numericValue()` dimension check and message
    /// (`IllegalStateException`).
    fn check_single(&self) -> Result<()> {
        if self.dims() != 1 {
            return Err(illegal(format!(
                "this field (name={}) uses {} dimensions; cannot convert to a single numeric value",
                self.name,
                self.dims()
            )));
        }
        Ok(())
    }
}

/// `IndexableField` for a sandbox point field: a packed point, not
/// inverted, with `numeric` as its `numericValue()`.
macro_rules! packed_indexable {
    ($ty:ident, $numeric:expr) => {
        impl lucene_index::document::IndexableField for $ty {
            fn name(&self) -> &str {
                &self.0.name
            }
            fn field_type(&self) -> &lucene_index::document::FieldType {
                &self.0.field_type
            }
            fn binary_value(&self) -> Option<std::borrow::Cow<'_, [u8]>> {
                Some(std::borrow::Cow::Borrowed(&self.0.packed))
            }
            fn numeric_value(&self) -> Option<lucene_index::document::Number> {
                #[allow(clippy::redundant_closure_call)]
                ($numeric)(self)
            }
            fn token_stream(
                &self,
                _analyzer: &lucene_analysis::Analyzer,
            ) -> lucene_index::document::Result<Option<lucene_index::document::FieldTokens>> {
                Ok(None)
            }
        }
    };
}
pub(crate) use packed_indexable;

#[cfg(test)]
mod tests {
    use lucene_analysis::Analyzer;
    use lucene_index::document::IndexableField;

    use super::*;

    #[test]
    fn sandbox_points_are_not_inverted() {
        let fields: [Box<dyn IndexableField>; 3] = [
            Box::new(HalfFloatPoint::new("h", &[1.5]).unwrap()),
            Box::new(BigIntegerPoint::new("b", &[7]).unwrap()),
            Box::new(LatLonBoundingBox::new("bb", -1.0, -2.0, 1.0, 2.0).unwrap()),
        ];
        for f in &fields {
            assert!(f.token_stream(&Analyzer::keyword()).unwrap().is_none());
            assert!(f.binary_value().is_some());
        }
    }
}
