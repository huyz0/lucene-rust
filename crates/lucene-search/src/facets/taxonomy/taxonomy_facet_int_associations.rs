//! `TaxonomyFacetIntAssociations`: aggregates the int values
//! [`IntAssociationFacetField`](super::IntAssociationFacetField)s indexed,
//! per ordinal.

use super::taxonomy_facets::{
    dense_slots, taxonomy_facets_wrapper, Aggregation, OrdValues, TaxonomyFacets,
};
use super::{AssociationAggregationFunction, TaxonomyReader};
use crate::facets::facet_utils::for_each_hit;
use crate::facets::facets_collector::FacetsCollector;
use crate::facets::{FacetsConfig, DEFAULT_INDEX_FIELD_NAME};
use crate::reader::doc_values::get_binary;
use crate::Result;

/// `TaxonomyFacetIntAssociations`. Derefs to its [`TaxonomyFacets`].
#[derive(Debug)]
pub struct TaxonomyFacetIntAssociations<'t> {
    inner: TaxonomyFacets<'t>,
}

taxonomy_facets_wrapper!(TaxonomyFacetIntAssociations);

impl<'t> TaxonomyFacetIntAssociations<'t> {
    /// `new TaxonomyFacetIntAssociations(taxoReader, config, fc,
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

    /// `new TaxonomyFacetIntAssociations(indexFieldName, taxoReader, config,
    /// fc, aggregationFunction)`. No roll-up: associations are flat.
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
            Aggregation::Int(aggregation_function),
        )?;
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            inner.initialize_value_counters();
            let mut dv = get_binary(hits.context.reader, index_field_name)?;
            if let (OrdValues::Dense(values), OrdValues::Dense(counts)) =
                (&mut inner.int_values, &mut inner.counts)
            {
                // Both arrays taken once: the same updates as below.
                for_each_hit(&hits.bits, &mut *dv, |dv| {
                    for_each_association(dv.binary_value(), |ord, bits| {
                        let (value, count) = dense_slots(values, counts, ord)?;
                        *value = aggregation_function.aggregate_int(*value, bits);
                        *count = count.wrapping_add(1);
                        Ok(())
                    })
                })?;
                continue;
            }
            for_each_hit(&hits.bits, &mut *dv, |dv| {
                for_each_association(dv.binary_value(), |ord, bits| {
                    inner.int_values.update(ord, |current| {
                        aggregation_function.aggregate_int(current, bits)
                    })?;
                    inner.counts.update(ord, |count| count.wrapping_add(1))
                })
            })?;
        }
        inner.finish_counting();
        Ok(TaxonomyFacetIntAssociations { inner })
    }
}

/// Each `(ordinal, value)` pair of an association field's bytes, both
/// big-endian ints (a float value is its bits). A trailing partial pair is
/// an error (Java reads past the value, into whatever the array holds).
pub(crate) fn for_each_association(
    bytes: &[u8],
    mut f: impl FnMut(i32, i32) -> Result<()>,
) -> Result<()> {
    let mut pairs = bytes.chunks(8);
    for pair in &mut pairs {
        let (Some(ord), Some(value)) = (be_int(pair, 0), be_int(pair, 4)) else {
            return Err(crate::Error::IllegalArgument(format!(
                "association bytes of length {} are not (ordinal, value) pairs",
                bytes.len()
            )));
        };
        f(ord, value)?;
    }
    Ok(())
}

/// The big-endian int at `at`.
#[inline]
fn be_int(bytes: &[u8], at: usize) -> Option<i32> {
    let b: [u8; 4] = bytes.get(at..at.checked_add(4)?)?.try_into().ok()?;
    Some(i32::from_be_bytes(b))
}

#[cfg(test)]
mod tests;
