//! `TaxonomyFacetFloatAssociations`: aggregates float values per ordinal --
//! the ones [`FloatAssociationFacetField`](super::FloatAssociationFacetField)s
//! indexed, or a [`DoubleValuesSource`]'s value of each matching document,
//! given to each of its ordinals.

use super::taxonomy_facet_int_associations::for_each_association;
use super::taxonomy_facets::{
    dense_slots, taxonomy_facets_wrapper, Aggregation, OrdValues, TaxonomyFacets,
};
use super::{AssociationAggregationFunction, TaxonomyReader};
use crate::facets::facet_utils::{for_each_hit, for_each_in_both};
use crate::facets::facets_collector::FacetsCollector;
use crate::facets::{FacetsConfig, DEFAULT_INDEX_FIELD_NAME};
use crate::reader::doc_values::{get_binary, get_sorted_numeric};
use crate::reader::SortedNumericDocValues;
use crate::values_source::{BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext};
use crate::Result;

/// `TaxonomyFacetFloatAssociations`. Derefs to its [`TaxonomyFacets`].
#[derive(Debug)]
pub struct TaxonomyFacetFloatAssociations<'t> {
    inner: TaxonomyFacets<'t>,
}

taxonomy_facets_wrapper!(TaxonomyFacetFloatAssociations);

impl<'t> TaxonomyFacetFloatAssociations<'t> {
    /// `new TaxonomyFacetFloatAssociations(taxoReader, config, fc,
    /// aggregationFunction)`: the default index field.
    pub fn new(
        taxo_reader: &'t dyn TaxonomyReader,
        config: &'t FacetsConfig,
        fc: &FacetsCollector<'_>,
        aggregation_function: AssociationAggregationFunction,
    ) -> Result<Self> {
        Self::with_field(
            DEFAULT_INDEX_FIELD_NAME,
            taxo_reader,
            config,
            fc,
            aggregation_function,
        )
    }

    /// `new TaxonomyFacetFloatAssociations(indexFieldName, taxoReader,
    /// config, fc, aggregationFunction)`: the indexed association values. No
    /// roll-up: associations are flat.
    pub fn with_field(
        index_field_name: &str,
        taxo_reader: &'t dyn TaxonomyReader,
        config: &'t FacetsConfig,
        fc: &FacetsCollector<'_>,
        aggregation_function: AssociationAggregationFunction,
    ) -> Result<Self> {
        let mut inner = TaxonomyFacets::new(
            index_field_name,
            taxo_reader,
            config,
            Some(fc),
            Aggregation::Float(aggregation_function),
        )?;
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            inner.initialize_value_counters();
            let mut dv = get_binary(hits.context.reader, index_field_name)?;
            if let (OrdValues::Dense(values), OrdValues::Dense(counts)) =
                (&mut inner.float_values, &mut inner.counts)
            {
                // Both arrays taken once: the same updates as below.
                for_each_hit(&hits.bits, &mut *dv, |dv| {
                    for_each_association(dv.binary_value(), |ord, bits| {
                        let (slot, count) = dense_slots(values, counts, ord)?;
                        *slot = aggregation_function
                            .aggregate_float(*slot, f32::from_bits(bits as u32));
                        *count = count.wrapping_add(1);
                        Ok(())
                    })
                })?;
                continue;
            }
            for_each_hit(&hits.bits, &mut *dv, |dv| {
                for_each_association(dv.binary_value(), |ord, bits| {
                    let value = f32::from_bits(bits as u32);
                    inner.float_values.update(ord, |current| {
                        aggregation_function.aggregate_float(current, value)
                    })?;
                    inner.counts.update(ord, |count| count.wrapping_add(1))
                })
            })?;
        }
        inner.finish_counting();
        Ok(TaxonomyFacetFloatAssociations { inner })
    }

    /// `new TaxonomyFacetFloatAssociations(taxoReader, config, fc,
    /// aggregationFunction, valuesSource)`: the default index field.
    pub fn with_values_source(
        taxo_reader: &'t dyn TaxonomyReader,
        config: &'t FacetsConfig,
        fc: &FacetsCollector<'_>,
        aggregation_function: AssociationAggregationFunction,
        values_source: &dyn DoubleValuesSource,
        ctx: &ValuesContext<'_>,
    ) -> Result<Self> {
        Self::with_field_and_values_source(
            DEFAULT_INDEX_FIELD_NAME,
            taxo_reader,
            config,
            fc,
            aggregation_function,
            values_source,
            ctx,
        )
    }

    /// `new TaxonomyFacetFloatAssociations(indexFieldName, taxoReader,
    /// config, fc, aggregationFunction, valuesSource)`: each matching
    /// document's value of `values_source` (read through `ctx`, the searcher
    /// `fc` collected from; given the kept scores when `fc` kept them) goes
    /// to each of its ordinals. Hierarchical dims are rolled up.
    pub fn with_field_and_values_source(
        index_field_name: &str,
        taxo_reader: &'t dyn TaxonomyReader,
        config: &'t FacetsConfig,
        fc: &FacetsCollector<'_>,
        aggregation_function: AssociationAggregationFunction,
        values_source: &dyn DoubleValuesSource,
        ctx: &ValuesContext<'_>,
    ) -> Result<Self> {
        let mut inner = TaxonomyFacets::new(
            index_field_name,
            taxo_reader,
            config,
            Some(fc),
            Aggregation::Float(aggregation_function),
        )?;
        let keep_scores = fc.keep_scores();
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            inner.initialize_value_counters();
            let mut ordinal_values = get_sorted_numeric(hits.context.reader, index_field_name)?;
            // The kept scores are borrowed, not copied, for the segment.
            let scores: Option<BoxDoubleValues<'_>> = if keep_scores {
                Some(Box::new(KeptScores {
                    scores: hits.scores.as_deref().unwrap_or_default(),
                    index: -1,
                }))
            } else {
                None
            };
            let mut function_values = values_source.get_values(ctx, hits.context.ord, scores)?;
            let mut it = hits.bits.iterator();
            let float_values = &mut inner.float_values;
            let counts = &mut inner.counts;
            for_each_in_both(&mut it, &mut *ordinal_values, |dv| {
                if !function_values.advance_exact(dv.doc_id())? {
                    return Ok(());
                }
                let value = function_values.double_value()? as f32;
                // Dense arrays (the usual case) matched once per document,
                // not twice per ordinal; the same updates either way.
                if let (OrdValues::Dense(values), OrdValues::Dense(counts)) =
                    (&mut *float_values, &mut *counts)
                {
                    return for_each_ordinal(dv, |ord| {
                        let (slot, count) = dense_slots(values, counts, ord)?;
                        *slot = aggregation_function.aggregate_float(*slot, value);
                        *count = count.wrapping_add(1);
                        Ok(())
                    });
                }
                for_each_ordinal(dv, |ord| {
                    float_values.update(ord, |current| {
                        aggregation_function.aggregate_float(current, value)
                    })?;
                    counts.update(ord, |count| count.wrapping_add(1))
                })
            })?;
        }
        // A values source can weigh hierarchical dims, so they roll up.
        inner.rollup()?;
        inner.finish_counting();
        Ok(TaxonomyFacetFloatAssociations { inner })
    }
}

/// `TaxonomyFacetFloatAssociations.scores(hits)`: the kept score of the
/// document advanced to (`0` past the last kept one).
struct KeptScores<'s> {
    scores: &'s [f32],
    index: i32,
}

impl DoubleValues for KeptScores<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.index = doc;
        Ok(true)
    }

    fn double_value(&mut self) -> Result<f64> {
        let score = usize::try_from(self.index)
            .ok()
            .and_then(|i| self.scores.get(i))
            .copied()
            .unwrap_or(0.0);
        Ok(f64::from(score))
    }
}

/// `f((int) value)` for each value of the current document, in bulk when
/// `dv` holds them decoded.
#[inline]
fn for_each_ordinal(
    dv: &mut (dyn SortedNumericDocValues + '_),
    mut f: impl FnMut(i32) -> Result<()>,
) -> Result<()> {
    if let Some(values) = dv.take_values() {
        for &value in values {
            f(value as i32)?;
        }
        return Ok(());
    }
    for _ in 0..dv.doc_value_count() {
        f(dv.next_value()? as i32)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
