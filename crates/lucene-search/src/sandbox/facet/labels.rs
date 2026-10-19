//! `org.apache.lucene.sandbox.facet.labels`: facet ordinals to labels and
//! back.

use crate::facets::range::DoubleRange;
use crate::facets::range::LongRange;
use crate::facets::taxonomy::facet_label::FacetLabel;
use crate::facets::taxonomy::taxonomy_reader::TaxonomyReader;
use crate::Result;

/// `LabelToOrd.INVALID_ORD`.
pub const INVALID_ORD: i32 = -1;

/// `OrdToLabel`: a facet ordinal's label; `None` where Java returns `null`
/// (an ordinal the mapping does not know).
pub trait OrdToLabel {
    /// `getLabel(ordinal)`.
    ///
    /// # Errors
    /// Whatever looking the label up reports.
    fn get_label(&self, ord: i32) -> Result<Option<FacetLabel>>;

    /// `getLabels(ordinals)`: each ordinal's label, in order.
    ///
    /// # Errors
    /// As [`get_label`](Self::get_label).
    fn get_labels(&self, ords: &[i32]) -> Result<Vec<Option<FacetLabel>>> {
        ords.iter().map(|&o| self.get_label(o)).collect()
    }
}

/// `LabelToOrd`: a label's facet ordinal, [`INVALID_ORD`] for an unknown
/// one.
pub trait LabelToOrd {
    /// `getOrd(label)`.
    ///
    /// # Errors
    /// Whatever looking the ordinal up reports.
    fn get_ord(&self, label: &FacetLabel) -> Result<i32>;

    /// `getOrds(labels)`.
    ///
    /// # Errors
    /// As [`get_ord`](Self::get_ord).
    fn get_ords(&self, labels: &[FacetLabel]) -> Result<Vec<i32>> {
        labels.iter().map(|l| self.get_ord(l)).collect()
    }
}

/// `RangeOrdToLabel`: a range facet's ordinal is the range's index; its
/// label is the range's.
#[derive(Debug, Clone)]
pub struct RangeOrdToLabel {
    labels: Vec<String>,
}

impl RangeOrdToLabel {
    /// `new RangeOrdToLabel(ranges)` from the ranges' labels, in order.
    pub fn new(labels: Vec<String>) -> Self {
        RangeOrdToLabel { labels }
    }

    /// `new RangeOrdToLabel(longRanges)`.
    pub fn for_long_ranges(ranges: &[LongRange]) -> Self {
        Self::new(ranges.iter().map(|r| r.label.clone()).collect())
    }

    /// `new RangeOrdToLabel(doubleRanges)`.
    pub fn for_double_ranges(ranges: &[DoubleRange]) -> Self {
        Self::new(ranges.iter().map(|r| r.label.clone()).collect())
    }
}

impl OrdToLabel for RangeOrdToLabel {
    fn get_label(&self, ord: i32) -> Result<Option<FacetLabel>> {
        match usize::try_from(ord).ok().and_then(|i| self.labels.get(i)) {
            Some(label) => FacetLabel::new([label.as_str()]).map(Some),
            None => Ok(None),
        }
    }
}

/// `TaxonomyOrdLabelBiMap`: the taxonomy's own ordinals and paths.
pub struct TaxonomyOrdLabelBiMap<'t> {
    taxo: &'t dyn TaxonomyReader,
}

impl<'t> TaxonomyOrdLabelBiMap<'t> {
    /// `new TaxonomyOrdLabelBiMap(taxoReader)`.
    pub fn new(taxo: &'t dyn TaxonomyReader) -> Self {
        TaxonomyOrdLabelBiMap { taxo }
    }
}

impl OrdToLabel for TaxonomyOrdLabelBiMap<'_> {
    fn get_label(&self, ord: i32) -> Result<Option<FacetLabel>> {
        self.taxo.get_path(ord).map(Some)
    }

    /// `taxoReader.getBulkPath(ordinals.clone())`.
    fn get_labels(&self, ords: &[i32]) -> Result<Vec<Option<FacetLabel>>> {
        Ok(self
            .taxo
            .get_bulk_path(ords)?
            .into_iter()
            .map(Some)
            .collect())
    }
}

impl LabelToOrd for TaxonomyOrdLabelBiMap<'_> {
    fn get_ord(&self, label: &FacetLabel) -> Result<i32> {
        self.taxo.get_ordinal(label)
    }

    fn get_ords(&self, labels: &[FacetLabel]) -> Result<Vec<i32>> {
        self.taxo.get_bulk_ordinals(labels)
    }
}
