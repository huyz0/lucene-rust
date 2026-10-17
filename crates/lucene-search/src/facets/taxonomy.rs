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

pub mod directory;
pub mod facet_label;
pub mod lru_hash_map;
pub mod parallel_taxonomy_arrays;
pub mod taxonomy_reader;
pub mod taxonomy_writer;
pub mod writercache;

pub use directory::{DirectoryTaxonomyReader, DirectoryTaxonomyWriter};
pub use facet_label::FacetLabel;
pub use lru_hash_map::LruHashMap;
pub use parallel_taxonomy_arrays::{IntArray, ParallelTaxonomyArrays};
pub use taxonomy_reader::{ChildrenIterator, TaxonomyReader, INVALID_ORDINAL, ROOT_ORDINAL};
pub use taxonomy_writer::TaxonomyWriter;
