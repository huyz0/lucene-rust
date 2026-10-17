//! `FastTaxonomyFacetCounts`: counts the taxonomy ordinals a field's
//! `SortedNumericDocValues` hold, over the hits of a [`FacetsCollector`] or
//! over every live document of an index.

use super::taxonomy_facets::{taxonomy_facets_wrapper, Aggregation, OrdValues, TaxonomyFacets};
use super::TaxonomyReader;
use crate::facets::facet_utils::for_each_hit;
use crate::facets::facets_collector::FacetsCollector;
use crate::facets::{FacetsConfig, DEFAULT_INDEX_FIELD_NAME};
use crate::reader::{IndexReader, SortedNumericDocValues, NO_MORE_DOCS};
use crate::{Error, Result};

/// `FastTaxonomyFacetCounts`. Derefs to its [`TaxonomyFacets`].
#[derive(Debug)]
pub struct FastTaxonomyFacetCounts<'t> {
    inner: TaxonomyFacets<'t>,
}

taxonomy_facets_wrapper!(FastTaxonomyFacetCounts);

impl<'t> FastTaxonomyFacetCounts<'t> {
    /// `new FastTaxonomyFacetCounts(taxoReader, config, fc)`: the default
    /// index field.
    pub fn new(
        taxo_reader: &'t dyn TaxonomyReader,
        config: &'t FacetsConfig,
        fc: &FacetsCollector<'_>,
    ) -> Result<Self> {
        Self::with_field(DEFAULT_INDEX_FIELD_NAME, taxo_reader, config, fc)
    }

    /// `new FastTaxonomyFacetCounts(indexFieldName, taxoReader, config, fc)`.
    pub fn with_field(
        index_field_name: &str,
        taxo_reader: &'t dyn TaxonomyReader,
        config: &'t FacetsConfig,
        fc: &FacetsCollector<'_>,
    ) -> Result<Self> {
        let mut inner = TaxonomyFacets::new(
            index_field_name,
            taxo_reader,
            config,
            Some(fc),
            Aggregation::Count,
        )?;
        count(&mut inner, fc)?;
        inner.rollup()?;
        inner.finish_counting();
        Ok(FastTaxonomyFacetCounts { inner })
    }

    /// `new FastTaxonomyFacetCounts(indexFieldName, reader, taxoReader,
    /// config)`: every live document of `reader`, as a `MatchAllDocsQuery`
    /// would count them.
    pub fn count_all(
        index_field_name: &str,
        reader: &dyn IndexReader,
        taxo_reader: &'t dyn TaxonomyReader,
        config: &'t FacetsConfig,
    ) -> Result<Self> {
        let mut inner = TaxonomyFacets::new(
            index_field_name,
            taxo_reader,
            config,
            None,
            Aggregation::Count,
        )?;
        for context in reader.leaves() {
            let Some(mut dv) = context.reader.sorted_numeric_doc_values(index_field_name)? else {
                continue;
            };
            inner.initialize_value_counters();
            let live_docs = context.reader.live_docs();
            // Without hits to size it by, the counting is always dense.
            let OrdValues::Dense(counts) = &mut inner.counts else {
                continue;
            };
            let mut doc = dv.next_doc()?;
            while doc != NO_MORE_DOCS {
                if live_docs.is_none_or(|live| live.get_doc(doc)) {
                    count_values(counts, &mut *dv)?;
                }
                doc = dv.next_doc()?;
            }
        }
        inner.rollup()?;
        inner.finish_counting();
        Ok(FastTaxonomyFacetCounts { inner })
    }
}

/// `counts[(int) value]++` for each value of the current document.
#[inline]
fn count_values(counts: &mut [i32], dv: &mut dyn SortedNumericDocValues) -> Result<()> {
    for_each_ordinal(dv, |ord| count_one(counts, ord))
}

/// `f((int) value)` for each value of the current document, in bulk when
/// `dv` holds them decoded.
#[inline]
fn for_each_ordinal(
    dv: &mut dyn SortedNumericDocValues,
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

/// `counts[ord]++`, refusing an ordinal past the taxonomy as Java's array
/// access does.
#[inline]
fn count_one(counts: &mut [i32], ord: i32) -> Result<()> {
    let len = counts.len();
    let slot = usize::try_from(ord)
        .ok()
        .and_then(|i| counts.get_mut(i))
        .ok_or_else(|| {
            Error::IllegalArgument(format!("Index {ord} out of bounds for length {len}"))
        })?;
    *slot = slot.wrapping_add(1);
    Ok(())
}

/// `count(matchingDocs)`.
fn count(facets: &mut TaxonomyFacets<'_>, fc: &FacetsCollector<'_>) -> Result<()> {
    let field = facets.index_field_name().to_string();
    for hits in fc.matching_docs() {
        if hits.total_hits == 0 {
            continue;
        }
        let Some(mut dv) = hits.context.reader.sorted_numeric_doc_values(&field)? else {
            continue;
        };
        facets.initialize_value_counters();
        // `initialize_value_counters` left the counts dense or sparse.
        if let OrdValues::Sparse(map) = &mut facets.counts {
            for_each_hit(&hits.bits, &mut *dv, |dv| {
                for_each_ordinal(dv, |ord| {
                    let slot = map.entry(ord).or_insert(0);
                    *slot = slot.wrapping_add(1);
                    Ok(())
                })
            })?;
        } else if let OrdValues::Dense(counts) = &mut facets.counts {
            for_each_hit(&hits.bits, &mut *dv, |dv| count_values(counts, dv))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
