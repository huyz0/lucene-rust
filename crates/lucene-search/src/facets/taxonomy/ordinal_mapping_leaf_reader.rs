//! `OrdinalMappingLeafReader`: a leaf whose facet ordinals -- the
//! SORTED_NUMERIC values of every facet index field of a [`FacetsConfig`] --
//! read through an ordinal map (one taxonomy's ordinals to another's, as
//! [`super::DirectoryTaxonomyWriter::add_taxonomy`] records them), each
//! document's values sorted again.
//!
//! It is a [`FilterLeafReader`] with an [`OrdinalMappingFilter`]: Java's
//! subclass is the filter, [`ordinal_mapping_leaf_reader`] its constructor.
//! [`super::taxonomy_merge_utils::merge`] does not read through it -- this
//! port's `addIndexes` merges segments, not readers -- but applies the same
//! map inside the merge
//! ([`lucene_index::merge::SortedNumericRemap`]).

use std::collections::HashSet;
use std::sync::Arc;

use crate::facets::{FacetsConfig, DEFAULT_INDEX_FIELD_NAME};
use crate::reader::filter::{AsDynLeaf, FilterLeafReader, LeafFilter};
use crate::reader::NO_MORE_DOCS;
use crate::reader::{DocIdSetIterator, DocValuesIterator, LeafReader, SortedNumericDocValues};
use crate::{Error, Result};

/// The facet index fields of `config`: every dim's, and the default's.
pub(crate) fn facet_fields(config: &FacetsConfig) -> HashSet<String> {
    let mut fields: HashSet<String> = config
        .dim_configs()
        .map(|(_, dc)| dc.index_field_name.clone())
        .collect();
    fields.insert(DEFAULT_INDEX_FIELD_NAME.to_string());
    fields
}

/// `OrdinalMappingLeafReader`'s overrides.
#[derive(Debug, Clone)]
pub struct OrdinalMappingFilter {
    ordinal_map: Arc<[i32]>,
    facet_fields: HashSet<String>,
}

impl OrdinalMappingFilter {
    /// The filter of `new OrdinalMappingLeafReader(in, ordinalMap,
    /// srcConfig)`.
    pub fn new(ordinal_map: Arc<[i32]>, src_config: &FacetsConfig) -> Self {
        OrdinalMappingFilter {
            ordinal_map,
            facet_fields: facet_fields(src_config),
        }
    }
}

impl LeafFilter for OrdinalMappingFilter {
    /// `getSortedNumericDocValues(field)`: a facet field's values mapped.
    fn wrap_sorted_numeric<'a>(
        &'a self,
        field: &str,
        values: Box<dyn SortedNumericDocValues + 'a>,
    ) -> Result<Box<dyn SortedNumericDocValues + 'a>> {
        if !self.facet_fields.contains(field) {
            return Ok(values);
        }
        Ok(Box::new(OrdinalMappingSortedNumericDocValues {
            in_: values,
            map: &self.ordinal_map[..],
            current: Vec::with_capacity(32),
            index: 0,
        }))
    }

    /// The mapped values are not the codec's singleton.
    fn keeps_sorted_numeric_singleton(&self, field: &str, inner: bool) -> bool {
        inner && !self.facet_fields.contains(field)
    }
}

/// `new OrdinalMappingLeafReader(in, ordinalMap, srcConfig)`.
pub fn ordinal_mapping_leaf_reader<R: ?Sized + LeafReader + AsDynLeaf>(
    in_: Arc<R>,
    ordinal_map: Arc<[i32]>,
    src_config: &FacetsConfig,
) -> FilterLeafReader<R> {
    FilterLeafReader::new(
        in_,
        Box::new(OrdinalMappingFilter::new(ordinal_map, src_config)),
    )
}

/// `OrdinalMappingSortedNumericDocValues`.
struct OrdinalMappingSortedNumericDocValues<'a> {
    in_: Box<dyn SortedNumericDocValues + 'a>,
    map: &'a [i32],
    /// `currentValues`: the current document's mapped values, sorted.
    current: Vec<i32>,
    /// `currIndex`.
    index: usize,
}

impl OrdinalMappingSortedNumericDocValues<'_> {
    /// `reloadValues()`.
    fn reload(&mut self) -> Result<()> {
        self.index = 0;
        self.current.clear();
        for _ in 0..self.in_.doc_value_count() {
            let original = self.in_.next_value()?;
            let mapped = usize::try_from(original)
                .ok()
                .and_then(|o| self.map.get(o))
                .ok_or_else(|| {
                    Error::IllegalState(format!(
                        "ordinal {original} is outside the ordinal map of {} ordinals",
                        self.map.len()
                    ))
                })?;
            self.current.push(*mapped);
        }
        self.current.sort_unstable();
        Ok(())
    }
}

impl DocIdSetIterator for OrdinalMappingSortedNumericDocValues<'_> {
    fn doc_id(&self) -> i32 {
        self.in_.doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        let doc = self.in_.next_doc()?;
        if doc != NO_MORE_DOCS {
            self.reload()?;
        }
        Ok(doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.in_.advance(target)?;
        if doc != NO_MORE_DOCS {
            self.reload()?;
        }
        Ok(doc)
    }

    fn cost(&self) -> i64 {
        self.in_.cost()
    }
}

impl DocValuesIterator for OrdinalMappingSortedNumericDocValues<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        let found = self.in_.advance_exact(target)?;
        if found {
            self.reload()?;
        }
        Ok(found)
    }
}

impl SortedNumericDocValues for OrdinalMappingSortedNumericDocValues<'_> {
    fn doc_value_count(&self) -> i32 {
        i32::try_from(self.current.len()).unwrap_or(i32::MAX)
    }

    /// `nextValue()`: past the last value, an error where Java asserts.
    fn next_value(&mut self) -> Result<i64> {
        let v = self.current.get(self.index).copied().ok_or_else(|| {
            Error::IllegalState("nextValue called past the document's values".into())
        })?;
        self.index = self.index.saturating_add(1);
        Ok(i64::from(v))
    }
}

#[cfg(test)]
mod tests;
