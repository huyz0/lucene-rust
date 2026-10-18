//! `TaxonomyFacetLabels`: the facet labels of a document, read back from
//! the taxonomy ordinals an index field holds.

use super::taxonomy_facets::parent_of;
use super::{FacetLabel, TaxonomyReader, INVALID_ORDINAL, ROOT_ORDINAL};
use crate::reader::doc_values::get_sorted_numeric;
use crate::reader::{LeafReaderContext, SortedNumericDocValues};
use crate::{Error, Result};

/// `TaxonomyFacetLabels`.
pub struct TaxonomyFacetLabels<'t> {
    taxo_reader: &'t dyn TaxonomyReader,
    index_field_name: String,
}

impl<'t> TaxonomyFacetLabels<'t> {
    /// `new TaxonomyFacetLabels(taxoReader, indexFieldName)`.
    pub fn new(taxo_reader: &'t dyn TaxonomyReader, index_field_name: &str) -> Self {
        TaxonomyFacetLabels {
            taxo_reader,
            index_field_name: index_field_name.to_string(),
        }
    }

    /// `getFacetLabelReader(readerContext)`: a leaf without the field reads
    /// no label.
    pub fn get_facet_label_reader<'l>(
        &self,
        context: LeafReaderContext<'l>,
    ) -> Result<FacetLabelReader<'t, 'l>> {
        Ok(FacetLabelReader {
            taxo_reader: self.taxo_reader,
            ordinal_values: get_sorted_numeric(context.reader, &self.index_field_name)?,
            current_doc_id: -1,
            current_doc_has_values: false,
            current_pos: 0,
            current_doc_ordinal_count: 0,
        })
    }
}

/// `TaxonomyFacetLabels.FacetLabelReader`: one leaf's labels, read document
/// by document in increasing order.
pub struct FacetLabelReader<'t, 'l> {
    taxo_reader: &'t dyn TaxonomyReader,
    ordinal_values: Box<dyn SortedNumericDocValues + 'l>,
    current_doc_id: i32,
    current_doc_has_values: bool,
    current_pos: i32,
    current_doc_ordinal_count: i32,
}

impl FacetLabelReader<'_, '_> {
    /// Moves to `doc_id` unless already there.
    fn advance_to(&mut self, doc_id: i32) -> Result<()> {
        if self.current_doc_id != doc_id {
            if doc_id < self.current_doc_id {
                return Err(Error::IllegalArgument(format!(
                    "docs out of order: previous docId={} current docId={doc_id}",
                    self.current_doc_id
                )));
            }
            self.current_doc_id = doc_id;
            self.current_doc_has_values = self.ordinal_values.advance_exact(doc_id)?;
            if self.current_doc_has_values {
                self.current_doc_ordinal_count = self.ordinal_values.doc_value_count();
                self.current_pos = 0;
            }
        }
        Ok(())
    }

    /// Whether the current document has a label left.
    fn has_next(&self) -> bool {
        self.current_doc_has_values && self.current_pos < self.current_doc_ordinal_count
    }

    /// The next ordinal of the current document.
    fn next_ordinal(&mut self) -> Result<i32> {
        let ord = self.ordinal_values.next_value()? as i32;
        self.current_pos = self.current_pos.saturating_add(1);
        Ok(ord)
    }

    /// `nextFacetLabel(docId)`: the document's next label, `None` once they
    /// are all read.
    pub fn next_facet_label(&mut self, doc_id: i32) -> Result<Option<FacetLabel>> {
        self.advance_to(doc_id)?;
        if !self.has_next() {
            return Ok(None);
        }
        let ord = self.next_ordinal()?;
        Ok(Some(self.taxo_reader.get_path(ord)?))
    }

    /// `nextFacetLabel(docId, facetDimension)`: the document's next label
    /// under `facet_dimension` (skipping the others), `None` once there is
    /// none.
    pub fn next_facet_label_in(
        &mut self,
        doc_id: i32,
        facet_dimension: &str,
    ) -> Result<Option<FacetLabel>> {
        let parent_ord = self
            .taxo_reader
            .get_ordinal(&FacetLabel::new([facet_dimension])?)?;
        if parent_ord == INVALID_ORDINAL {
            return Err(Error::IllegalArgument(format!(
                "Category ordinal not found for facet dimension: {facet_dimension}"
            )));
        }
        self.advance_to(doc_id)?;
        if !self.has_next() {
            return Ok(None);
        }
        let arrays = self.taxo_reader.parallel_taxonomy_arrays()?;
        let parents = arrays.parents();
        loop {
            let ord = self.next_ordinal()?;
            // `isDescendant(ord, parentOrd)`.
            let mut o = ord;
            while o != INVALID_ORDINAL && o != ROOT_ORDINAL {
                let parent = parent_of(parents, o)?;
                if parent == parent_ord {
                    return Ok(Some(self.taxo_reader.get_path(ord)?));
                }
                o = parent;
            }
            if self.current_pos >= self.current_doc_ordinal_count {
                return Ok(None);
            }
        }
    }
}

#[cfg(test)]
mod tests;
