//! `org.apache.lucene.sandbox.facet.cutters`: a document's taxonomy
//! ordinals ([`TaxonomyFacetsCutter`]) or its distinct long values
//! ([`LongValueFacetCutter`]) as facet ordinals.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use crate::directory_reader::SegmentReader;
use crate::facets::taxonomy::facet_label::FacetLabel;
use crate::facets::taxonomy::parallel_taxonomy_arrays::ParallelTaxonomyArrays;
use crate::facets::taxonomy::taxonomy_reader::TaxonomyReader;
use crate::facets::FacetsConfig;
use crate::reader::doc_values::get_sorted_numeric;
use crate::reader::SortedNumericDocValues;
use crate::{Error, Result};

use super::labels::{OrdToLabel, INVALID_ORD};
use super::{
    from_array, BoxOrdinalIterator, FacetCutter, FastMap, LeafFacetCutter, OrdinalIterator,
    NO_MORE_ORDS,
};

/// `TaxonomyFacetsCutter`: a document's taxonomy ordinals, read from the
/// index field's `SORTED_NUMERIC` doc values (repeats kept); dims that are
/// not multi-valued are rolled up.
pub struct TaxonomyFacetsCutter<'t> {
    config: &'t FacetsConfig,
    taxo: &'t dyn TaxonomyReader,
    index_field_name: String,
    disable_rollup: bool,
    arrays: OnceLock<Arc<dyn ParallelTaxonomyArrays>>,
}

impl<'t> TaxonomyFacetsCutter<'t> {
    /// `new TaxonomyFacetsCutter(indexFieldName, facetsConfig, taxoReader)`.
    pub fn new(
        index_field_name: impl Into<String>,
        config: &'t FacetsConfig,
        taxo: &'t dyn TaxonomyReader,
    ) -> Self {
        Self::with_rollup(index_field_name, config, taxo, false)
    }

    /// `new TaxonomyFacetsCutter(indexFieldName, facetsConfig, taxoReader,
    /// disableRollup)`.
    pub fn with_rollup(
        index_field_name: impl Into<String>,
        config: &'t FacetsConfig,
        taxo: &'t dyn TaxonomyReader,
        disable_rollup: bool,
    ) -> Self {
        TaxonomyFacetsCutter {
            config,
            taxo,
            index_field_name: index_field_name.into(),
            disable_rollup,
            arrays: OnceLock::new(),
        }
    }

    /// `getChildren()`/`getSiblings()`: the taxonomy's arrays, read once.
    fn arrays(&self) -> Result<Arc<dyn ParallelTaxonomyArrays>> {
        if let Some(a) = self.arrays.get() {
            return Ok(a.clone());
        }
        let a = self.taxo.parallel_taxonomy_arrays()?;
        Ok(self.arrays.get_or_init(|| a).clone())
    }
}

impl FacetCutter for TaxonomyFacetsCutter<'_> {
    fn create_leaf_cutter<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetCutter + 'a>> {
        let values = get_sorted_numeric(reader, &self.index_field_name)?;
        Ok(Box::new(TaxonomyLeafFacetCutter {
            values,
            ords_in_doc: 0,
        }))
    }

    /// The ordinals of the configured dims that are not multi-valued and
    /// live in this cutter's index field (hierarchical or not), skipping
    /// dims the taxonomy never saw; `None` with rollup disabled.
    fn ordinals_to_rollup(&self) -> Result<Option<BoxOrdinalIterator<'_>>> {
        if self.disable_rollup {
            return Ok(None);
        }
        let dims = self
            .config
            .dim_configs()
            .filter(|(_, c)| !c.multi_valued && c.index_field_name == self.index_field_name)
            .map(|(dim, _)| FacetLabel::new([dim]))
            .collect::<Result<Vec<_>>>()?;
        let ords = self.taxo.get_bulk_ordinals(&dims)?;
        Ok(Some(Box::new(from_array(
            ords.into_iter().filter(|&o| o != INVALID_ORD).collect(),
        ))))
    }

    fn children_ords(&self, ord: i32) -> Result<Option<BoxOrdinalIterator<'_>>> {
        Ok(Some(Box::new(ChildrenOrdinalIterator {
            arrays: self.arrays()?,
            parent: ord,
            current: ord,
        })))
    }
}

/// `TaxonomyFacetsCutter.getChildrenOrds`' iterator: the first child, then
/// its siblings.
struct ChildrenOrdinalIterator {
    arrays: Arc<dyn ParallelTaxonomyArrays>,
    parent: i32,
    current: i32,
}

impl OrdinalIterator for ChildrenOrdinalIterator {
    fn next_ord(&mut self) -> Result<i32> {
        if self.current == NO_MORE_ORDS {
            return Ok(NO_MORE_ORDS);
        }
        self.current = if self.current == self.parent {
            self.arrays.children().get(self.current)
        } else {
            self.arrays.siblings().get(self.current)
        };
        Ok(if self.current == INVALID_ORD {
            NO_MORE_ORDS
        } else {
            self.current
        })
    }
}

/// `TaxonomyFacetsCutter.TaxonomyLeafFacetCutterMultiValue`.
struct TaxonomyLeafFacetCutter<'a> {
    values: Box<dyn SortedNumericDocValues + 'a>,
    ords_in_doc: i32,
}

impl OrdinalIterator for TaxonomyLeafFacetCutter<'_> {
    fn next_ord(&mut self) -> Result<i32> {
        if self.ords_in_doc > 0 {
            self.ords_in_doc -= 1;
            // `(int) nextValue()`.
            return Ok(self.values.next_value()? as i32);
        }
        Ok(NO_MORE_ORDS)
    }
}

impl LeafFacetCutter for TaxonomyLeafFacetCutter<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        if self.values.advance_exact(doc)? {
            self.ords_in_doc = self.values.doc_value_count();
            return Ok(true);
        }
        Ok(false)
    }
}

/// The value-to-ordinal map shared by a [`LongValueFacetCutter`] and its
/// leaf cutters.
struct LongValueOrds {
    value_to_ord: RwLock<FastMap<i64, i32>>,
    /// `maxOrdinal`: the last ordinal handed out.
    max_ordinal: AtomicI32,
}

impl LongValueOrds {
    /// `valueToOrdMap.computeIfAbsent(value, maxOrdinal::incrementAndGet)`.
    fn ord_of(&self, value: i64) -> Result<i32> {
        if let Some(&ord) = self.read()?.get(&value) {
            return Ok(ord);
        }
        let mut map = self
            .value_to_ord
            .write()
            .map_err(|_| Error::IllegalState("facet cutter state poisoned".into()))?;
        Ok(*map.entry(value).or_insert_with(|| {
            self.max_ordinal
                .fetch_add(1, Ordering::SeqCst)
                .wrapping_add(1)
        }))
    }

    fn read(&self) -> Result<std::sync::RwLockReadGuard<'_, FastMap<i64, i32>>> {
        self.value_to_ord
            .read()
            .map_err(|_| Error::IllegalState("facet cutter state poisoned".into()))
    }
}

/// `LongValueFacetCutter`: a document's distinct values of a long field,
/// each value's ordinal handed out at its first sight (one ordinal space
/// over all segments).
pub struct LongValueFacetCutter {
    field: String,
    ords: Arc<LongValueOrds>,
    /// `ordToValueMap`: built at the first [`get_value`](Self::get_value) or
    /// label, and never again (as Java's).
    ord_to_value: OnceLock<FastMap<i32, i64>>,
}

impl LongValueFacetCutter {
    /// `new LongValueFacetCutter(field)`.
    pub fn new(field: impl Into<String>) -> Self {
        let ords = LongValueOrds {
            value_to_ord: RwLock::default(),
            max_ordinal: AtomicI32::new(-1),
        };
        LongValueFacetCutter {
            field: field.into(),
            ords: Arc::new(ords),
            ord_to_value: OnceLock::new(),
        }
    }

    /// The field.
    pub fn field(&self) -> &str {
        &self.field
    }

    fn ord_to_value(&self) -> Result<&FastMap<i32, i64>> {
        if let Some(m) = self.ord_to_value.get() {
            return Ok(m);
        }
        let built: FastMap<i32, i64> = self.ords.read()?.iter().map(|(&v, &o)| (o, v)).collect();
        Ok(self.ord_to_value.get_or_init(|| built))
    }

    /// `getValue(ordinal)`: 0 for an unknown ordinal.
    ///
    /// # Errors
    /// [`Error::IllegalState`] for a poisoned map.
    pub fn get_value(&self, ord: i32) -> Result<i64> {
        Ok(self.ord_to_value()?.get(&ord).copied().unwrap_or(0))
    }
}

impl OrdToLabel for LongValueFacetCutter {
    /// The value as a one-component label; `None` for an unknown ordinal
    /// (Java's assertion, then `null`).
    fn get_label(&self, ord: i32) -> Result<Option<FacetLabel>> {
        match self.ord_to_value()?.get(&ord) {
            Some(v) => FacetLabel::new([v.to_string()]).map(Some),
            None => Ok(None),
        }
    }
}

impl FacetCutter for LongValueFacetCutter {
    fn create_leaf_cutter<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetCutter + 'a>> {
        Ok(Box::new(LongValueLeafFacetCutter {
            values: get_sorted_numeric(reader, &self.field)?,
            ords: self.ords.clone(),
            seen: FastMap::default(),
            count: 0,
            cursor: 0,
            last: 0,
        }))
    }
}

/// `LongValueFacetCutter.createLeafCutter`'s cutter: a document's values,
/// a run of equal ones counted once.
struct LongValueLeafFacetCutter<'a> {
    values: Box<dyn SortedNumericDocValues + 'a>,
    ords: Arc<LongValueOrds>,
    /// The ordinals this leaf has already looked up: an ordinal never
    /// changes once handed out, so the shared map's lock is taken once per
    /// distinct value per segment rather than once per value.
    seen: FastMap<i64, i32>,
    count: i32,
    cursor: i32,
    last: i64,
}

impl OrdinalIterator for LongValueLeafFacetCutter<'_> {
    fn next_ord(&mut self) -> Result<i32> {
        while self.cursor < self.count {
            self.cursor += 1;
            let value = self.values.next_value()?;
            // Sorted values: only the previous value can repeat this one.
            if self.cursor == 1 || value != self.last {
                self.last = value;
                if let Some(&ord) = self.seen.get(&value) {
                    return Ok(ord);
                }
                let ord = self.ords.ord_of(value)?;
                self.seen.insert(value, ord);
                return Ok(ord);
            }
        }
        Ok(NO_MORE_ORDS)
    }
}

impl LeafFacetCutter for LongValueLeafFacetCutter<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        if self.values.advance_exact(doc)? {
            self.count = self.values.doc_value_count();
            self.cursor = 0;
            return Ok(true);
        }
        Ok(false)
    }
}
