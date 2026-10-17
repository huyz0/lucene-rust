//! `AssociationAggregationFunction`: how the association facets combine the
//! values of one ordinal.

use crate::facets::facets_collector::java_max;

/// `AssociationAggregationFunction`: `MAX` or `SUM` (Lucene has no other).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssociationAggregationFunction {
    /// `MAX`: `Math.max` (a float `NaN` wins, `0.0` beats `-0.0`).
    Max,
    /// `SUM`: Java's `+` (an int sum wraps on overflow).
    Sum,
}

impl AssociationAggregationFunction {
    /// `aggregate(int existingVal, int newVal)`.
    #[inline]
    pub fn aggregate_int(self, existing: i32, new: i32) -> i32 {
        match self {
            AssociationAggregationFunction::Max => existing.max(new),
            AssociationAggregationFunction::Sum => existing.wrapping_add(new),
        }
    }

    /// `aggregate(float existingVal, float newVal)`.
    #[inline]
    pub fn aggregate_float(self, existing: f32, new: f32) -> f32 {
        match self {
            AssociationAggregationFunction::Max => java_max(existing, new),
            AssociationAggregationFunction::Sum => existing + new,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AssociationAggregationFunction::{Max, Sum};

    #[test]
    fn int_sum_wraps_and_max_picks_the_greater() {
        assert_eq!(Sum.aggregate_int(2, 3), 5);
        assert_eq!(Sum.aggregate_int(i32::MAX, 1), i32::MIN);
        assert_eq!(Max.aggregate_int(-4, -7), -4);
        assert_eq!(Max.aggregate_int(i32::MIN, 0), 0);
    }

    #[test]
    fn float_max_is_math_max() {
        assert_eq!(Sum.aggregate_float(1.5, 2.25), 3.75);
        assert_eq!(Max.aggregate_float(-1.0, -3.0), -1.0);
        assert!(Max.aggregate_float(f32::NAN, 1.0).is_nan());
        assert!(Max.aggregate_float(1.0, f32::NAN).is_nan());
        assert!(Max.aggregate_float(-0.0, 0.0).is_sign_positive());
        assert!(Max.aggregate_float(0.0, -0.0).is_sign_positive());
        assert!(Max.aggregate_float(-0.0, -0.0).is_sign_negative());
    }
}
