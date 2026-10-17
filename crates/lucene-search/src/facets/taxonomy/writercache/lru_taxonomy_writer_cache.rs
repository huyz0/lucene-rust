//! `LruTaxonomyWriterCache`: a [`TaxonomyWriterCache`] holding at most a
//! fixed number of categories, dropping the least recently used third when
//! it overflows.

use super::super::FacetLabel;
use super::name_int_cache_lru::{NameHashIntCacheLru, NameIntCacheLru};
use super::TaxonomyWriterCache;

/// `LruTaxonomyWriterCache.LRUType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LruType {
    /// `LRU_HASHED`: keyed by each label's 64-bit hash -- less memory, and
    /// two labels with the same hash collide.
    LruHashed,
    /// `LRU_STRING`: keyed by the label. The default.
    #[default]
    LruString,
}

#[derive(Debug)]
enum Cache {
    String(NameIntCacheLru),
    Hashed(NameHashIntCacheLru),
}

/// `LruTaxonomyWriterCache`.
#[derive(Debug)]
pub struct LruTaxonomyWriterCache {
    /// `None` once closed (Java nulls the field, and a later call throws a
    /// `NullPointerException`; here a closed cache answers as an empty one).
    cache: Option<Cache>,
}

impl LruTaxonomyWriterCache {
    /// `new LruTaxonomyWriterCache(cacheSize)`: [`LruType::LruString`].
    pub fn new(cache_size: i32) -> Self {
        Self::with_type(cache_size, LruType::LruString)
    }

    /// `new LruTaxonomyWriterCache(cacheSize, lruType)`.
    pub fn with_type(cache_size: i32, lru_type: LruType) -> Self {
        let cache = match lru_type {
            LruType::LruHashed => Cache::Hashed(NameHashIntCacheLru::new(cache_size)),
            LruType::LruString => Cache::String(NameIntCacheLru::new(cache_size)),
        };
        LruTaxonomyWriterCache { cache: Some(cache) }
    }
}

impl TaxonomyWriterCache for LruTaxonomyWriterCache {
    fn close(&mut self) {
        self.clear();
        self.cache = None;
    }

    fn get(&mut self, category_path: &FacetLabel) -> i32 {
        let res = match &mut self.cache {
            Some(Cache::String(c)) => c.get(category_path),
            Some(Cache::Hashed(c)) => c.get(category_path),
            None => None,
        };
        res.unwrap_or(-1)
    }

    fn put(&mut self, category_path: &FacetLabel, ordinal: i32) -> bool {
        match &mut self.cache {
            Some(Cache::String(c)) => {
                let ret = c.put(category_path, ordinal);
                if ret {
                    c.make_room_lru();
                }
                ret
            }
            Some(Cache::Hashed(c)) => {
                let ret = c.put(category_path, ordinal);
                if ret {
                    c.make_room_lru();
                }
                ret
            }
            None => false,
        }
    }

    fn is_full(&self) -> bool {
        let (size, max) = match &self.cache {
            Some(Cache::String(c)) => (c.size(), c.max_size()),
            Some(Cache::Hashed(c)) => (c.size(), c.max_size()),
            None => return false,
        };
        i64::try_from(size).unwrap_or(i64::MAX) == i64::from(max)
    }

    fn clear(&mut self) {
        match &mut self.cache {
            Some(Cache::String(c)) => c.clear(),
            Some(Cache::Hashed(c)) => c.clear(),
            None => {}
        }
    }

    fn size(&self) -> usize {
        match &self.cache {
            Some(Cache::String(c)) => c.size(),
            Some(Cache::Hashed(c)) => c.size(),
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests;
