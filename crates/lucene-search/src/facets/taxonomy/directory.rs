//! `org.apache.lucene.facet.taxonomy.directory`: the taxonomy kept in a
//! Lucene index in a [`lucene_store::Directory`].

pub mod consts;
pub mod directory_taxonomy_reader;
pub mod directory_taxonomy_writer;
pub mod taxonomy_index_arrays;

pub use directory_taxonomy_reader::DirectoryTaxonomyReader;
pub use directory_taxonomy_writer::{
    DirectoryTaxonomyWriter, DiskOrdinalMap, MemoryOrdinalMap, OrdinalMap,
};
pub use taxonomy_index_arrays::TaxonomyIndexArrays;
