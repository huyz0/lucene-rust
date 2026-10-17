//! `org.apache.lucene.facet.taxonomy`: the taxonomy index (M13 T13.4a) --
//! a separate Lucene index with one document per category, whose document
//! id is the category's ordinal, so a main index can record a document's
//! categories as small integers (`FacetsConfig.build`, in
//! [`crate::facets::FacetsConfig::build`]).
//!
//! The on-disk format is Lucene's: each category document carries a
//! `NumericDocValuesField("$parent_ndv$", parent)`, a
//! `BinaryDocValuesField("$full_path$", path)` and an indexed, unstored
//! `StringField("$full_path$", path)`, the path being
//! [`crate::facets::path_components_to_string`]'s encoding; the commit user
//! data carries the taxonomy's epoch (`index.epoch`, in hexadecimal).
//! Lucene reads a taxonomy this port writes and the other way round
//! (`fixtures/src/GenTaxonomy.java`, `fixtures/src/VerifyTaxonomy.java`).
//!
//! Lucene 10.5.0 has no `index.create.time` and no
//! `InconsistentTaxonomyException`: a reader that finds the epoch changed
//! under it (the taxonomy was recreated) simply reopens as a fresh reader,
//! sharing nothing with the old one.
//!
//! The counting side (M13 T13.4b): [`FastTaxonomyFacetCounts`] counts a
//! search's hits per ordinal, [`TaxonomyFacetIntAssociations`] and
//! [`TaxonomyFacetFloatAssociations`] aggregate the values
//! [`AssociationFacetField`]s indexed (or a values source's), all on
//! [`TaxonomyFacets`]; [`TaxonomyFacetLabels`] reads a document's labels
//! back.

pub mod association_aggregation_function;
pub mod association_facet_field;
pub mod directory;
pub mod facet_label;
pub mod fast_taxonomy_facet_counts;
pub mod lru_hash_map;
pub mod parallel_taxonomy_arrays;
pub mod taxonomy_facet_float_associations;
pub mod taxonomy_facet_int_associations;
pub mod taxonomy_facet_labels;
pub mod taxonomy_facets;
pub mod taxonomy_reader;
pub mod taxonomy_writer;
pub mod writercache;

pub use association_aggregation_function::AssociationAggregationFunction;
pub use association_facet_field::{
    AssociationFacetField, AssociationKind, FloatAssociationFacetField, IntAssociationFacetField,
};
pub use directory::{DirectoryTaxonomyReader, DirectoryTaxonomyWriter};
pub use facet_label::FacetLabel;
pub use fast_taxonomy_facet_counts::FastTaxonomyFacetCounts;
pub use lru_hash_map::LruHashMap;
pub use parallel_taxonomy_arrays::{IntArray, ParallelTaxonomyArrays};
pub use taxonomy_facet_float_associations::TaxonomyFacetFloatAssociations;
pub use taxonomy_facet_int_associations::TaxonomyFacetIntAssociations;
pub use taxonomy_facet_labels::{FacetLabelReader, TaxonomyFacetLabels};
pub use taxonomy_facets::{Aggregation, TaxonomyFacets};
pub use taxonomy_reader::{ChildrenIterator, TaxonomyReader, INVALID_ORDINAL, ROOT_ORDINAL};
pub use taxonomy_writer::TaxonomyWriter;
