//! `IntPointMultiRangeBuilder`, `LongPointMultiRangeBuilder`,
//! `FloatPointMultiRangeBuilder` and `DoublePointMultiRangeBuilder`: a
//! [`MultiRangeQuery`] over the matching `*Point` field, each range checked
//! against the builder's dimensions and for a lower bound above its upper
//! one.

use lucene_index::document::{DoublePoint, FloatPoint, IntPoint, LongPoint};

use super::illegal;
use crate::sandbox::search::{MultiRangeQuery, MultiRangeQueryBuilder};
use crate::Result;

macro_rules! multi_range_builder {
    ($(#[$m:meta])* $name:ident, $point:ident, $prim:ty) => {
        $(#[$m])*
        #[derive(Debug, Clone)]
        pub struct $name(MultiRangeQueryBuilder);

        impl $name {
            /// `new XxxPointMultiRangeBuilder(field, numDims)`.
            pub fn new(field: impl Into<String>, num_dims: usize) -> Result<Self> {
                MultiRangeQueryBuilder::new(field, $point::BYTES, num_dims).map($name)
            }

            /// `add(lowerValue[], upperValue[])`.
            pub fn add(&mut self, lower: &[$prim], upper: &[$prim]) -> Result<&mut Self> {
                let n = self.0.num_dims;
                if upper.len() != n || lower.len() != n {
                    return Err(illegal(
                        "Passed in range does not conform to specified dimensions",
                    ));
                }
                if lower.iter().zip(upper).any(|(lo, hi)| hi < lo) {
                    return Err(illegal(
                        "Upper value of range should be greater than lower value of range",
                    ));
                }
                let lo = $point::pack(lower).map_err(|e| illegal(e.to_string()))?;
                let hi = $point::pack(upper).map_err(|e| illegal(e.to_string()))?;
                self.0.add(lo, hi);
                Ok(self)
            }

            /// `MultiRangeQuery.Builder.add(byte[], byte[])`: packed bounds,
            /// unchecked.
            pub fn add_packed(&mut self, lower: Vec<u8>, upper: Vec<u8>) -> &mut Self {
                self.0.add(lower, upper);
                self
            }

            /// `build()`.
            pub fn build(&self) -> MultiRangeQuery {
                self.0.build()
            }
        }
    };
}

multi_range_builder!(
    /// `IntPointMultiRangeBuilder`.
    IntPointMultiRangeBuilder, IntPoint, i32
);
multi_range_builder!(
    /// `LongPointMultiRangeBuilder`.
    LongPointMultiRangeBuilder, LongPoint, i64
);
multi_range_builder!(
    /// `FloatPointMultiRangeBuilder` (`upper < lower` is `false` for `NaN`,
    /// so a `NaN` bound passes the check, as in Java).
    FloatPointMultiRangeBuilder, FloatPoint, f32
);
multi_range_builder!(
    /// `DoublePointMultiRangeBuilder`.
    DoublePointMultiRangeBuilder, DoublePoint, f64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_and_builds() {
        let mut b = IntPointMultiRangeBuilder::new("i", 2).unwrap();
        let e = b.add(&[1], &[2]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "document query: Passed in range does not conform to specified dimensions"
        );
        let e = b.add(&[1, 5], &[2, 4]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "document query: Upper value of range should be greater than lower value of range"
        );
        b.add(&[1, 2], &[3, 4])
            .unwrap()
            .add_packed(vec![0; 8], vec![1; 8]);
        let q = b.build();
        assert_eq!(
            (q.num_dims, q.bytes_per_dim, q.range_clauses.len()),
            (2, 4, 2)
        );
        assert!(IntPointMultiRangeBuilder::new("i", 0).is_err());
        let mut l = LongPointMultiRangeBuilder::new("l", 1).unwrap();
        l.add(&[i64::MIN], &[0]).unwrap();
        assert_eq!(l.build().bytes_per_dim, 8);
        let mut f = FloatPointMultiRangeBuilder::new("f", 1).unwrap();
        f.add(&[1.0], &[f32::NAN]).unwrap();
        assert!(f.add(&[2.0], &[1.0]).is_err());
        let mut d = DoublePointMultiRangeBuilder::new("d", 1).unwrap();
        d.add(&[-1.0], &[1.0]).unwrap();
        assert_eq!(d.build().range_clauses.len(), 1);
    }
}
