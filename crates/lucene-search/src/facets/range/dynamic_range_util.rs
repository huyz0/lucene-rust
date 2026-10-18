//! `DynamicRangeUtil`: ranges chosen from the data so that each holds about
//! the same total weight.
//!
//! Differs: Java reads the segments on an `ExecutorService`; here they are
//! read in turn. The outcome is the same -- a segment whose weights
//! overflow fails first (Java waits on every segment before merging), then
//! the overflow of the total.

use crate::facets::facet_utils::for_each_doc;
use crate::facets::facets_collector::FacetsCollector;
use crate::values_source::{LongValuesSource, ValuesContext};
use crate::{Error, Result};

/// `DynamicRangeUtil.DynamicRangeInfo`: one range -- how many values and how
/// much weight it holds, its smallest and largest value, and the mean value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicRangeInfo {
    /// `count`.
    pub count: i32,
    /// `weight`.
    pub weight: i64,
    /// `min`.
    pub min: i64,
    /// `max`.
    pub max: i64,
    /// `centroid`.
    pub centroid: f64,
}

/// `computeDynamicRanges(weightFieldName, weightValueSource,
/// fieldValueSource, facetsCollector, topN, exec)`: the matching documents'
/// values of `field_value_source`, weighted by `weight_value_source` (a
/// document without either, or with weight `0`, is skipped), cut into at
/// most `top_n` ranges. Both sources are read through `ctx`, a context over
/// the searcher the hits came from.
///
/// # Errors
/// [`Error::IllegalArgument`] when a segment's or the overall total weight
/// overflows a `long`; whatever reading the values reports.
pub fn compute_dynamic_ranges(
    weight_field_name: &str,
    weight_value_source: &dyn LongValuesSource,
    field_value_source: &dyn LongValuesSource,
    facets_collector: &FacetsCollector<'_>,
    top_n: i32,
    ctx: &ValuesContext<'_>,
) -> Result<Vec<DynamicRangeInfo>> {
    // [1] each segment's values and weights.
    let mut segments: Vec<(Vec<i64>, Vec<i64>, i64)> = Vec::new();
    for hits in facets_collector.matching_docs() {
        if hits.total_hits <= 0 {
            continue;
        }
        let mut field_value = field_value_source.get_values(ctx, hits.context.ord, None)?;
        let mut weight_value = weight_value_source.get_values(ctx, hits.context.ord, None)?;
        let mut values = Vec::new();
        let mut weights = Vec::new();
        let mut total = 0i64;
        for_each_doc(&hits.bits, |doc| {
            if !field_value.advance_exact(doc)? || !weight_value.advance_exact(doc)? {
                return Ok(());
            }
            let value = field_value.long_value()?;
            let weight = weight_value.long_value()?;
            if weight == 0 {
                return Ok(());
            }
            values.push(value);
            weights.push(weight);
            total = total.checked_add(weight).ok_or_else(|| {
                Error::IllegalArgument("segment long totalWeight value out of bounds".into())
            })?;
            Ok(())
        })?;
        segments.push((values, weights, total));
    }
    // [3] merged, with the total weight.
    let mut values = Vec::new();
    let mut weights = Vec::new();
    let mut total_weight = 0i64;
    for (v, w, total) in segments {
        total_weight = total.checked_add(total_weight).ok_or_else(|| {
            Error::IllegalArgument(format!(
                "weight field \"{weight_field_name}\": long totalWeight value out of bounds"
            ))
        })?;
        values.extend(v);
        weights.extend(w);
    }
    let len = values.len();
    Ok(compute_dynamic_numeric_ranges(
        &mut values,
        &mut weights,
        len,
        total_weight,
        top_n,
    ))
}

/// `computeDynamicNumericRanges(values, weights, len, totalWeight, topN)`:
/// the first `len` values (sorted in place, ties by weight, with their
/// weights), cut each time the weight gathered reaches
/// `total_weight / min(top_n, len)`; what is left over is the last range.
/// A negative `top_n`, which Java asserts against, gives no ranges.
///
/// # Panics
/// When `len` exceeds either slice's length (Java asserts it does not).
pub fn compute_dynamic_numeric_ranges(
    values: &mut [i64],
    weights: &mut [i64],
    len: usize,
    total_weight: i64,
    top_n: i32,
) -> Vec<DynamicRangeInfo> {
    let mut result = Vec::new();
    if len == 0 || top_n <= 0 {
        return result;
    }
    // A total order on (value, weight), so any sort gives Java's order.
    let mut pairs: Vec<(i64, i64)> = values[..len]
        .iter()
        .copied()
        .zip(weights[..len].iter().copied())
        .collect();
    pairs.sort_unstable();
    for (i, &(v, w)) in pairs.iter().enumerate() {
        values[i] = v;
        weights[i] = w;
    }

    let mut accu_weight = 0i64;
    let mut value_sum = 0i64;
    let mut count = 0i32;
    let mut min_idx = 0usize;
    let ranges = (top_n as usize).min(len);
    let range_weight_target = total_weight as f64 / ranges as f64;
    for i in 0..len {
        accu_weight = accu_weight.wrapping_add(weights[i]);
        value_sum = value_sum.wrapping_add(values[i]);
        count = count.wrapping_add(1);
        if accu_weight as f64 >= range_weight_target {
            result.push(DynamicRangeInfo {
                count,
                weight: accu_weight,
                min: values[min_idx],
                max: values[i],
                centroid: value_sum as f64 / f64::from(count),
            });
            count = 0;
            accu_weight = 0;
            value_sum = 0;
            min_idx = i + 1;
        }
    }
    if min_idx < len {
        result.push(DynamicRangeInfo {
            count,
            weight: accu_weight,
            min: values[min_idx],
            max: values[len - 1],
            centroid: value_sum as f64 / f64::from(count),
        });
    }
    result
}

#[cfg(test)]
mod tests;
