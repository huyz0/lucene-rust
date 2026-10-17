//! `org.apache.lucene.facet.taxonomy.writercache`: the category -> ordinal
//! caches a [`super::directory::DirectoryTaxonomyWriter`] consults before it
//! looks a category up in its index.

pub mod char_block_array;
pub mod label_to_ordinal;
pub mod lru_taxonomy_writer_cache;
pub mod name_int_cache_lru;

pub use char_block_array::CharBlockArray;
pub use label_to_ordinal::LabelToOrdinal;
pub use lru_taxonomy_writer_cache::{LruTaxonomyWriterCache, LruType};
pub use name_int_cache_lru::{NameHashIntCacheLru, NameIntCacheLru};

use super::FacetLabel;

/// `TaxonomyWriterCache`. Java's implementations synchronize each method;
/// here the writer owns its cache behind its own lock, so the methods take
/// `&mut self`.
pub trait TaxonomyWriterCache: Send + std::fmt::Debug {
    /// `close()`: let go of the cache's resources; a closed cache is empty
    /// and stays empty.
    fn close(&mut self);
    /// `get(categoryPath)`: the ordinal, or a negative number when absent.
    fn get(&mut self, category_path: &FacetLabel) -> i32;
    /// `put(categoryPath, ordinal)`: whether entries were evicted to make
    /// room (which means [`TaxonomyWriterCache::get`] may now miss
    /// categories that exist).
    fn put(&mut self, category_path: &FacetLabel, ordinal: i32) -> bool;
    /// `isFull()`: whether a further `put` would evict.
    fn is_full(&self) -> bool;
    /// `clear()`.
    fn clear(&mut self);
    /// `size()`.
    fn size(&self) -> usize;
}
