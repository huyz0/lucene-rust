//! `FacetField`, `SortedSetDocValuesFacetField` and `FacetsConfig.build`:
//! a document's facet labels turned into the fields that get indexed.
//!
//! Java finds the facet fields among the document's fields by their
//! `FieldType` instance; here they are passed beside the document, as
//! [`FacetEntry`]s in document order. The result is the same document Java
//! builds: the taxonomy fields (per index field, in Java's `HashMap` order),
//! then the sorted-set fields, then the association fields (likewise), then
//! every other field of `doc` in its order.

use std::collections::HashSet;

use lucene_index::document::{
    BinaryDocValuesField, Document, SortedNumericDocValuesField, SortedSetDocValuesField, Store,
    StringField,
};

use super::taxonomy::{AssociationFacetField, FacetLabel, TaxonomyWriter};
use super::{DimConfig, DrillDownTermsIndexing, FacetsConfig};
use crate::{Error, Result};

/// `FacetField.verifyLabel(label)`.
pub fn verify_label(label: &str) -> Result<()> {
    if label.is_empty() {
        return Err(Error::IllegalArgument(format!(
            "empty or null components not allowed; got: {label}"
        )));
    }
    Ok(())
}

/// `FacetField`: a taxonomy facet label, `dim` plus a non-empty `path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetField {
    pub dim: String,
    pub path: Vec<String>,
}

impl FacetField {
    /// `new FacetField(dim, path...)`.
    pub fn new<S: AsRef<str>>(dim: &str, path: &[S]) -> Result<Self> {
        verify_label(dim)?;
        for label in path {
            verify_label(label.as_ref())?;
        }
        if path.is_empty() {
            return Err(Error::IllegalArgument(
                "path must have at least one element".to_string(),
            ));
        }
        Ok(FacetField {
            dim: dim.to_string(),
            path: path.iter().map(|s| s.as_ref().to_string()).collect(),
        })
    }
}

impl std::fmt::Display for FacetField {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FacetField(dim={} path=[{}])",
            self.dim,
            self.path.join(", ")
        )
    }
}

/// `SortedSetDocValuesFacetField`: a sorted-set facet label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedSetDocValuesFacetField {
    pub dim: String,
    pub path: Vec<String>,
}

impl SortedSetDocValuesFacetField {
    /// `new SortedSetDocValuesFacetField(dim, path...)`: the path's labels
    /// are checked before the dim, as in Java.
    pub fn new<S: AsRef<str>>(dim: &str, path: &[S]) -> Result<Self> {
        for label in path {
            verify_label(label.as_ref())?;
        }
        verify_label(dim)?;
        if path.is_empty() {
            return Err(Error::IllegalArgument(
                "path must have at least one element".to_string(),
            ));
        }
        Ok(SortedSetDocValuesFacetField {
            dim: dim.to_string(),
            path: path.iter().map(|s| s.as_ref().to_string()).collect(),
        })
    }
}

/// One facet field of a document handed to [`FacetsConfig::build`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FacetEntry {
    /// A [`FacetField`] (taxonomy).
    Taxonomy(FacetField),
    /// A [`SortedSetDocValuesFacetField`].
    SortedSet(SortedSetDocValuesFacetField),
    /// An [`AssociationFacetField`] (taxonomy, with an associated value).
    Association(AssociationFacetField),
}

impl From<AssociationFacetField> for FacetEntry {
    fn from(f: AssociationFacetField) -> Self {
        FacetEntry::Association(f)
    }
}

impl From<FacetField> for FacetEntry {
    fn from(f: FacetField) -> Self {
        FacetEntry::Taxonomy(f)
    }
}

impl From<SortedSetDocValuesFacetField> for FacetEntry {
    fn from(f: SortedSetDocValuesFacetField) -> Self {
        FacetEntry::SortedSet(f)
    }
}

/// `computeIfAbsent(indexFieldName, ...).add(field)` into a list of groups
/// kept in insertion order.
fn group<'f, T>(groups: &mut Vec<(String, Vec<&'f T>)>, key: &str, field: &'f T) {
    match groups.iter_mut().find(|(k, _)| k == key) {
        Some((_, fields)) => fields.push(field),
        None => groups.push((key.to_string(), vec![field])),
    }
}

/// A `HashMap<String, List<...>>`'s entries in Java's iteration order.
fn in_java_order<'g, T>(groups: &'g [(String, Vec<&'g T>)]) -> Vec<&'g (String, Vec<&'g T>)> {
    let keys: Vec<&str> = groups.iter().map(|(k, _)| k.as_str()).collect();
    crate::mlt::java_hash_order(&keys)
        .into_iter()
        .map(|i| &groups[i])
        .collect()
}

/// `checkTaxoWriter(taxoWriter)`.
fn check_taxo_writer(taxo_writer: Option<&dyn TaxonomyWriter>) -> Result<&dyn TaxonomyWriter> {
    taxo_writer.ok_or_else(|| {
        Error::IllegalState(
            "a non-null TaxonomyWriter must be provided when indexing FacetField or \
             AssociationFacetField"
                .to_string(),
        )
    })
}

fn not_hierarchical(dim: &str, components: usize) -> Error {
    Error::IllegalArgument(format!(
        "dimension \"{dim}\" is not hierarchical yet has {components} components"
    ))
}

impl FacetsConfig {
    /// `FacetsConfig.build(taxoWriter, doc)`: `doc` with `facets` translated
    /// into the fields that get indexed (see the module docs for the order).
    /// `taxo_writer` may be `None` when `facets` holds no
    /// [`FacetEntry::Taxonomy`] or [`FacetEntry::Association`].
    pub fn build(
        &self,
        taxo_writer: Option<&dyn TaxonomyWriter>,
        facets: &[FacetEntry],
        doc: Document,
    ) -> Result<Document> {
        let mut by_field: Vec<(String, Vec<&FacetField>)> = Vec::new();
        let mut dv_by_field: Vec<(String, Vec<&SortedSetDocValuesFacetField>)> = Vec::new();
        let mut assoc_by_field: Vec<(String, Vec<&AssociationFacetField>)> = Vec::new();
        let mut seen_dims: HashSet<&str> = HashSet::new();
        for entry in facets {
            let dim = match entry {
                FacetEntry::Taxonomy(f) => &f.dim,
                FacetEntry::SortedSet(f) => &f.dim,
                FacetEntry::Association(f) => &f.dim,
            };
            let config = self.dim_config(dim);
            if !config.multi_valued && !seen_dims.insert(dim) {
                // `checkSeen`.
                return Err(Error::IllegalArgument(format!(
                    "dimension \"{dim}\" is not multiValued, but it appears more than once in \
                     this document"
                )));
            }
            match entry {
                FacetEntry::Taxonomy(f) => group(&mut by_field, &config.index_field_name, f),
                FacetEntry::SortedSet(f) => group(&mut dv_by_field, &config.index_field_name, f),
                FacetEntry::Association(f) => {
                    if config.hierarchical {
                        return Err(Error::IllegalArgument(format!(
                            "AssociationFacetField cannot be hierarchical (dim=\"{dim}\")"
                        )));
                    }
                    if config.require_dim_count {
                        return Err(Error::IllegalArgument(format!(
                            "AssociationFacetField cannot requireDimCount (dim=\"{dim}\")"
                        )));
                    }
                    let index_field_name = &config.index_field_name;
                    group(&mut assoc_by_field, index_field_name, f);
                    // Best effort: the types indexed into one field must agree.
                    let kind = f.kind().type_name();
                    let current = self.assoc_dim_types.get_or_record(index_field_name, kind);
                    if current != kind {
                        return Err(Error::IllegalArgument(format!(
                            "mixing incompatible types of AssociationFacetField ({current} and \
                             {kind}) in indexed field \"{index_field_name}\"; use FacetsConfig to \
                             change the indexFieldName for each dimension"
                        )));
                    }
                }
            }
        }

        let mut result = Document::new();
        self.process_facet_fields(taxo_writer, &by_field, &mut result)?;
        self.process_ssdv_facet_fields(&dv_by_field, &mut result)?;
        self.process_assoc_facet_fields(taxo_writer, &assoc_by_field, &mut result)?;
        for field in doc.into_fields() {
            result.add_boxed(field);
        }
        Ok(result)
    }

    /// `processFacetFields(taxoWriter, byField, doc)`.
    fn process_facet_fields(
        &self,
        taxo_writer: Option<&dyn TaxonomyWriter>,
        by_field: &[(String, Vec<&FacetField>)],
        doc: &mut Document,
    ) -> Result<()> {
        for (index_field_name, fields) in in_java_order(by_field) {
            let mut ordinals: Vec<i32> = Vec::new();
            for facet_field in fields {
                let config = self.dim_config(&facet_field.dim);
                if facet_field.path.len() > 1 && !config.hierarchical {
                    return Err(not_hierarchical(&facet_field.dim, facet_field.path.len()));
                }
                let label = FacetLabel::from_dim_path(&facet_field.dim, &facet_field.path)?;
                let taxo_writer = check_taxo_writer(taxo_writer)?;
                let ordinal = taxo_writer.add_category(&label)?;
                ordinals.push(ordinal);
                if config.multi_valued && (config.hierarchical || config.require_dim_count) {
                    // Every ancestor too, the root excluded.
                    let mut parent = taxo_writer.get_parent(ordinal)?;
                    while parent > 0 {
                        ordinals.push(parent);
                        parent = taxo_writer.get_parent(parent)?;
                    }
                    if !config.require_dim_count {
                        // The last one is the dim's.
                        ordinals.pop();
                    }
                }
                index_drill_down_terms(doc, index_field_name, config, &label);
            }
            // Sorted to drop duplicates (`SortedNumericDocValuesField` keeps
            // them).
            ordinals.sort_unstable();
            let mut prev = -1;
            for ord in ordinals {
                if ord > prev {
                    doc.add(SortedNumericDocValuesField::new(
                        index_field_name.as_str(),
                        i64::from(ord),
                    ));
                    prev = ord;
                }
            }
        }
        Ok(())
    }

    /// `processAssocFacetFields(taxoWriter, byField, doc)`: per index field,
    /// one binary doc-values field of big-endian `(ordinal, value)` pairs (no
    /// parents), and each label's drill-down terms.
    fn process_assoc_facet_fields(
        &self,
        taxo_writer: Option<&dyn TaxonomyWriter>,
        by_field: &[(String, Vec<&AssociationFacetField>)],
        doc: &mut Document,
    ) -> Result<()> {
        for (index_field_name, fields) in in_java_order(by_field) {
            let mut bytes: Vec<u8> = Vec::with_capacity(16);
            for field in fields {
                let taxo_writer = check_taxo_writer(taxo_writer)?;
                let label = FacetLabel::from_dim_path(&field.dim, &field.path)?;
                let ordinal = taxo_writer.add_category(&label)?;
                bytes.extend_from_slice(&ordinal.to_be_bytes());
                bytes.extend_from_slice(&field.assoc);
                index_drill_down_terms(doc, index_field_name, self.dim_config(&field.dim), &label);
            }
            doc.add(BinaryDocValuesField::new(index_field_name.as_str(), bytes));
        }
        Ok(())
    }

    /// `processSSDVFacetFields(byField, doc)`.
    fn process_ssdv_facet_fields(
        &self,
        by_field: &[(String, Vec<&SortedSetDocValuesFacetField>)],
        doc: &mut Document,
    ) -> Result<()> {
        for (index_field_name, fields) in in_java_order(by_field) {
            for facet_field in fields {
                let label = FacetLabel::from_dim_path(&facet_field.dim, &facet_field.path)?;
                let config = self.dim_config(&facet_field.dim);
                if config.hierarchical {
                    // Every prefix, so every unique path has an ordinal.
                    for i in 1..=label.len() {
                        doc.add(SortedSetDocValuesField::new(
                            index_field_name.as_str(),
                            label.subpath(i).to_term(),
                        ));
                    }
                } else {
                    if label.len() != 2 {
                        return Err(not_hierarchical(&facet_field.dim, facet_field.path.len()));
                    }
                    if config.multi_valued && config.require_dim_count {
                        doc.add(SortedSetDocValuesField::new(
                            index_field_name.as_str(),
                            facet_field.dim.as_bytes(),
                        ));
                    }
                    doc.add(SortedSetDocValuesField::new(
                        index_field_name.as_str(),
                        label.to_term(),
                    ));
                }
                index_drill_down_terms(doc, index_field_name, config, &label);
            }
        }
        Ok(())
    }
}

/// `indexDrillDownTerms(doc, indexFieldName, dimConfig, facetLabel)`.
fn index_drill_down_terms(
    doc: &mut Document,
    index_field_name: &str,
    config: &DimConfig,
    label: &FacetLabel,
) {
    let indexing = config.drill_down_terms_indexing;
    if indexing == DrillDownTermsIndexing::None {
        return;
    }
    doc.add(StringField::new(
        index_field_name,
        label.to_term(),
        Store::No,
    ));
    let prefixes = match indexing {
        DrillDownTermsIndexing::None | DrillDownTermsIndexing::FullPathOnly => 0..0,
        DrillDownTermsIndexing::DimensionAndFullPath => 1..2,
        DrillDownTermsIndexing::AllPathsNoDim => 2..label.len(),
        DrillDownTermsIndexing::All => 1..label.len(),
    };
    for i in prefixes {
        doc.add(StringField::new(
            index_field_name,
            label.subpath(i).to_term(),
            Store::No,
        ));
    }
}

#[cfg(test)]
mod tests;
