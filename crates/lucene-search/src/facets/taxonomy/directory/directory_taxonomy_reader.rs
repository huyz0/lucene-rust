//! `DirectoryTaxonomyReader`: a [`TaxonomyReader`] over a taxonomy index,
//! opened from its directory (a commit) or from its
//! [`DirectoryTaxonomyWriter`] (near-real-time).
//!
//! As in Java, ordinal and path lookups go through two LRU caches of 4000
//! entries ([`DirectoryTaxonomyReader::set_cache_size`]), which a reopened
//! reader shares with the one it came from unless the taxonomy was
//! recreated (its epoch changed): a recreated taxonomy reuses ordinals for
//! other categories, so the reopened reader starts empty.
//!
//! Differs from Java: the reader borrows its directory (or writer) for its
//! lifetime, and the index files it reads stay open until it is dropped --
//! [`TaxonomyReader::close`] makes it refuse further calls but cannot
//! release what the borrow checker still lets other readers share.
//! `toString()` of the internal reader, quoted in the out-of-range error,
//! names each segment as `name:maxDoc[/deletes]` (this port's segment
//! readers do not carry the writing Lucene version or the compound flag
//! Java prints).

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use lucene_store::Directory;

use super::consts::FULL;
use super::directory_taxonomy_writer::{DirectoryTaxonomyWriter, TaxoNrtSource, INDEX_EPOCH};
use super::taxonomy_index_arrays::TaxonomyIndexArrays;
use crate::directory_reader::DirectoryReader;
use crate::facets::taxonomy::lru_hash_map::LruHashMap;
use crate::facets::taxonomy::parallel_taxonomy_arrays::ParallelTaxonomyArrays;
use crate::facets::taxonomy::taxonomy_reader::{RefCount, INVALID_ORDINAL, ROOT_ORDINAL};
use crate::facets::taxonomy::{FacetLabel, TaxonomyReader};
use crate::reader::{IndexReader, LeafReaderContext, PostingsFlags, NO_MORE_DOCS};
use crate::{Error, Result};

/// `DEFAULT_CACHE_VALUE`.
const DEFAULT_CACHE_VALUE: usize = 4000;

/// `BYTES_PER_CACHE_ENTRY`: `4 * NUM_BYTES_OBJECT_REF + 4 *
/// NUM_BYTES_OBJECT_HEADER + 8 * Character.BYTES` (compressed oops).
const BYTES_PER_CACHE_ENTRY: i64 = 4 * 4 + 4 * 12 + 8 * 2;

type OrdinalCache = Arc<Mutex<LruHashMap<FacetLabel, i32>>>;
type CategoryCache = Arc<Mutex<LruHashMap<i32, FacetLabel>>>;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn new_ordinal_cache() -> OrdinalCache {
    Arc::new(Mutex::new(LruHashMap::new(DEFAULT_CACHE_VALUE)))
}

fn new_category_cache() -> CategoryCache {
    Arc::new(Mutex::new(LruHashMap::new(DEFAULT_CACHE_VALUE)))
}

/// `DirectoryTaxonomyReader`.
pub struct DirectoryTaxonomyReader<'a> {
    dir: &'a dyn Directory,
    /// `taxoWriter`: set for a near-real-time reader.
    taxo_writer: Option<&'a dyn TaxoNrtSource>,
    /// `taxoEpoch`: the writer's epoch when this reader was opened, `-1`
    /// without a writer.
    taxo_epoch: i64,
    index_reader: DirectoryReader,
    ordinal_cache: OrdinalCache,
    category_cache: CategoryCache,
    taxo_arrays: OnceLock<Arc<TaxonomyIndexArrays>>,
    ref_count: RefCount,
}

impl std::fmt::Debug for DirectoryTaxonomyReader<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectoryTaxonomyReader")
            .field("index_reader", &self.reader_description())
            .finish_non_exhaustive()
    }
}

impl<'a> DirectoryTaxonomyReader<'a> {
    /// `new DirectoryTaxonomyReader(directory)`: the latest commit.
    pub fn open(dir: &'a dyn Directory) -> Result<Self> {
        let index_reader = DirectoryReader::open(dir)?;
        Ok(Self::from_parts(
            dir,
            index_reader,
            None,
            new_ordinal_cache(),
            new_category_cache(),
            None,
        ))
    }

    /// `new DirectoryTaxonomyReader(taxoWriter)`: everything the writer has
    /// added, committed or not.
    pub fn from_writer<'d: 'a>(taxo_writer: &'a DirectoryTaxonomyWriter<'d>) -> Result<Self> {
        // Java reads the epoch before it opens the reader: a recreate in
        // between leaves this reader stamped with the old epoch, so the next
        // `openIfChanged` sees the recreate rather than sharing its caches.
        let taxo_epoch = taxo_writer.taxonomy_epoch();
        let index_reader = taxo_writer.open_nrt_reader()?;
        let mut reader = Self::from_parts(
            taxo_writer.directory(),
            index_reader,
            Some(taxo_writer as &dyn TaxoNrtSource),
            new_ordinal_cache(),
            new_category_cache(),
            None,
        );
        reader.taxo_epoch = taxo_epoch;
        Ok(reader)
    }

    /// The package-private constructor: `taxo_arrays`, when given, is grown
    /// over the new reader rather than recomputed.
    fn from_parts(
        dir: &'a dyn Directory,
        index_reader: DirectoryReader,
        taxo_writer: Option<&'a dyn TaxoNrtSource>,
        ordinal_cache: OrdinalCache,
        category_cache: CategoryCache,
        taxo_arrays: Option<TaxonomyIndexArrays>,
    ) -> Self {
        let arrays = OnceLock::new();
        if let Some(a) = taxo_arrays {
            let _ = arrays.set(Arc::new(a));
        }
        DirectoryTaxonomyReader {
            dir,
            taxo_writer,
            taxo_epoch: taxo_writer.map_or(-1, |w| w.taxonomy_epoch()),
            index_reader,
            ordinal_cache,
            category_cache,
            taxo_arrays: arrays,
            ref_count: RefCount::default(),
        }
    }

    /// `TaxonomyReader.openIfChanged(reader)` / `doOpenIfChanged()`: `None`
    /// when nothing changed, else a new reader -- sharing this one's caches
    /// and arrays unless the taxonomy was recreated.
    pub fn open_if_changed(&self) -> Result<Option<Self>> {
        self.ensure_open()?;
        let r2 = match self.taxo_writer {
            Some(w) => w.reopen_nrt_reader(&self.index_reader)?,
            None => self.index_reader.open_if_changed(self.dir)?,
        };
        let Some(r2) = r2 else {
            return Ok(None);
        };
        let recreated = match self.taxo_writer {
            None => {
                let t1 = epoch_of(&self.index_reader);
                let t2 = epoch_of(&r2);
                match t1 {
                    None => t2.is_some(),
                    Some(t1) => Some(t1) != t2,
                }
            }
            Some(w) => self.taxo_epoch != w.taxonomy_epoch(),
        };
        if recreated {
            return Ok(Some(Self::from_parts(
                self.dir,
                r2,
                self.taxo_writer,
                new_ordinal_cache(),
                new_category_cache(),
                None,
            )));
        }
        let arrays = match self.taxo_arrays.get() {
            Some(old) => Some(TaxonomyIndexArrays::copy_from(&r2, old)?),
            None => None,
        };
        Ok(Some(Self::from_parts(
            self.dir,
            r2,
            self.taxo_writer,
            Arc::clone(&self.ordinal_cache),
            Arc::clone(&self.category_cache),
            arrays,
        )))
    }

    /// `getInternalIndexReader()`.
    pub fn internal_index_reader(&self) -> Result<&DirectoryReader> {
        self.ensure_open()?;
        Ok(&self.index_reader)
    }

    /// `setCacheSize(size)`: both caches' capacity.
    pub fn set_cache_size(&self, size: usize) -> Result<()> {
        self.ensure_open()?;
        lock(&self.category_cache).set_max_size(size);
        lock(&self.ordinal_cache).set_max_size(size);
        Ok(())
    }

    /// `ramBytesUsed()`: the arrays' shallow size and a fixed estimate per
    /// cache entry.
    pub fn ram_bytes_used(&self) -> Result<i64> {
        self.ensure_open()?;
        let mut ram = self.taxo_arrays.get().map_or(0, |a| a.ram_bytes_used());
        let entries =
            |n: usize| BYTES_PER_CACHE_ENTRY.saturating_mul(i64::try_from(n).unwrap_or(i64::MAX));
        ram = ram.saturating_add(entries(lock(&self.category_cache).len()));
        ram = ram.saturating_add(entries(lock(&self.ordinal_cache).len()));
        Ok(ram)
    }

    /// `toString(max)`: the first `max` categories, one per line.
    pub fn to_string(&self, max: usize) -> Result<String> {
        self.ensure_open()?;
        let mut sb = String::new();
        let limit = max.min(usize::try_from(self.index_reader.max_doc()).unwrap_or(0));
        for i in 0..limit {
            let ordinal = i32::try_from(i).unwrap_or(i32::MAX);
            match self.get_path(ordinal) {
                Ok(category) if category.is_empty() => {
                    sb.push_str(&format!("{i}: EMPTY STRING!! \n"));
                }
                Ok(category) => sb.push_str(&format!("{i}: {category}\n")),
                Err(Error::Store(_)) => sb.push_str(&format!("{i}: EXCEPTION!! \n")),
                Err(e) => return Err(e),
            }
        }
        Ok(sb)
    }

    /// `indexReader.toString()`, best effort (see the module docs).
    fn reader_description(&self) -> String {
        let infos = &self.index_reader.segment_infos;
        let mut s = format!(
            "StandardDirectoryReader(segments_{}:{}",
            lucene_util::base36::to_base36(infos.generation),
            infos.version
        );
        if self.taxo_writer.is_some() {
            s.push_str(":nrt");
        }
        for seg in &infos.segments {
            s.push(' ');
            s.push_str(&seg.segment_name);
            s.push(':');
            let max_doc = self
                .index_reader
                .segment_readers()
                .iter()
                .find(|r| r.segment_name == seg.segment_name)
                .map_or(0, |r| r.max_doc);
            s.push_str(&max_doc.to_string());
            if seg.del_count != 0 {
                s.push_str(&format!("/{}", seg.del_count));
            }
        }
        s.push(')');
        s
    }

    /// `checkOrdinalBounds(ordinals...)`.
    fn check_ordinal_bounds(&self, ordinals: &[i32]) -> Result<()> {
        let max_doc = self.index_reader.max_doc();
        for &ordinal in ordinals {
            if ordinal < 0 || ordinal >= max_doc {
                return Err(Error::IllegalArgument(format!(
                    "ordinal {ordinal} is out of the range of the indexReader {}. The maximum \
                     possible ordinal number is {}",
                    self.reader_description(),
                    max_doc.saturating_sub(1)
                )));
            }
        }
        Ok(())
    }

    /// `ReaderUtil.subIndex(ordinal, leaves)`: the last leaf starting at or
    /// before `ordinal`.
    fn leaf_of<'r>(
        leaves: &[LeafReaderContext<'r>],
        ordinal: i32,
    ) -> Option<LeafReaderContext<'r>> {
        let at = leaves.partition_point(|l| l.doc_base <= ordinal);
        at.checked_sub(1).and_then(|i| leaves.get(i)).copied()
    }

    /// The label stored for `ordinal` in `leaf`'s `$full_path$` doc values.
    fn read_path(leaf: &LeafReaderContext<'_>, ordinal: i32) -> Result<FacetLabel> {
        let missing = || Error::IllegalState(format!("no {FULL} doc value for ordinal {ordinal}"));
        let mut values = leaf.reader.binary_doc_values(FULL)?.ok_or_else(missing)?;
        if !values.advance_exact(ordinal.wrapping_sub(leaf.doc_base))? {
            return Err(missing());
        }
        FacetLabel::from_term(values.binary_value())
    }
}

/// The commit's [`INDEX_EPOCH`].
fn epoch_of(reader: &DirectoryReader) -> Option<&str> {
    reader
        .segment_infos
        .user_data
        .iter()
        .rev()
        .find(|(k, _)| k == INDEX_EPOCH)
        .map(|(_, v)| v.as_str())
}

impl TaxonomyReader for DirectoryTaxonomyReader<'_> {
    fn ref_counter(&self) -> &RefCount {
        &self.ref_count
    }

    fn do_close(&self) -> Result<()> {
        Ok(())
    }

    fn parallel_taxonomy_arrays(&self) -> Result<Arc<dyn ParallelTaxonomyArrays>> {
        self.ensure_open()?;
        if let Some(a) = self.taxo_arrays.get() {
            return Ok(Arc::clone(a) as Arc<dyn ParallelTaxonomyArrays>);
        }
        let arrays = Arc::new(TaxonomyIndexArrays::new(&self.index_reader)?);
        let arrays = self.taxo_arrays.get_or_init(|| arrays);
        Ok(Arc::clone(arrays) as Arc<dyn ParallelTaxonomyArrays>)
    }

    fn commit_user_data(&self) -> Result<Vec<(String, String)>> {
        self.ensure_open()?;
        Ok(self.index_reader.segment_infos.user_data.clone())
    }

    fn get_ordinal(&self, cp: &FacetLabel) -> Result<i32> {
        self.ensure_open()?;
        if cp.is_empty() {
            return Ok(ROOT_ORDINAL);
        }
        let max_doc = self.index_reader.max_doc();
        if let Some(&res) = lock(&self.ordinal_cache).get(cp) {
            return Ok(if res < max_doc { res } else { INVALID_ORDINAL });
        }
        let term = cp.to_term();
        let mut ret = INVALID_ORDINAL;
        for leaf in self.index_reader.leaves() {
            let Some(terms) = leaf.reader.terms(FULL)? else {
                continue;
            };
            let mut te = terms.iterator()?;
            if te.try_seek_exact(term.as_bytes())? {
                let mut docs = te.postings(PostingsFlags::None)?;
                let doc = docs.next_doc()?;
                if doc != NO_MORE_DOCS {
                    ret = doc.wrapping_add(leaf.doc_base);
                    break;
                }
            }
        }
        if ret != INVALID_ORDINAL {
            lock(&self.ordinal_cache).put(cp.clone(), ret);
        }
        Ok(ret)
    }

    fn get_bulk_ordinals(&self, category_paths: &[FacetLabel]) -> Result<Vec<i32>> {
        self.ensure_open()?;
        match category_paths {
            [] => return Ok(Vec::new()),
            [one] => return Ok(vec![self.get_ordinal(one)?]),
            _ => {}
        }
        let max_doc = self.index_reader.max_doc();
        let mut result = vec![INVALID_ORDINAL; category_paths.len()];
        let mut missing: Vec<usize> = Vec::new();
        {
            let mut cache = lock(&self.ordinal_cache);
            for (i, cp) in category_paths.iter().enumerate() {
                match cache.get(cp) {
                    Some(&res) => result[i] = if res < max_doc { res } else { INVALID_ORDINAL },
                    None => missing.push(i),
                }
            }
        }
        if missing.is_empty() {
            return Ok(result);
        }
        let terms_to_get: Vec<String> = missing
            .iter()
            .map(|&i| category_paths[i].to_term())
            .collect();
        let mut order: Vec<usize> = (0..missing.len()).collect();
        order.sort_by(|&a, &b| terms_to_get[a].as_bytes().cmp(terms_to_get[b].as_bytes()));
        // The merged terms enum's first document of each term is the first
        // leaf's (in document order) that has the term.
        let mut found = vec![false; missing.len()];
        for leaf in self.index_reader.leaves() {
            let Some(terms) = leaf.reader.terms(FULL)? else {
                continue;
            };
            let mut te = terms.iterator()?;
            for &k in &order {
                if found[k] || !te.try_seek_exact(terms_to_get[k].as_bytes())? {
                    continue;
                }
                let mut docs = te.postings(PostingsFlags::None)?;
                let doc = docs.next_doc()?;
                if doc != NO_MORE_DOCS {
                    result[missing[k]] = doc.wrapping_add(leaf.doc_base);
                    found[k] = true;
                }
            }
        }
        let mut cache = lock(&self.ordinal_cache);
        for &k in &order {
            let res_index = missing[k];
            let ord = result[res_index];
            if ord != INVALID_ORDINAL {
                cache.put(category_paths[res_index].clone(), ord);
            }
        }
        Ok(result)
    }

    fn get_path(&self, ordinal: i32) -> Result<FacetLabel> {
        self.ensure_open()?;
        self.check_ordinal_bounds(&[ordinal])?;
        if let Some(label) = lock(&self.category_cache).get(&ordinal) {
            return Ok(label.clone());
        }
        let leaves = self.index_reader.leaves();
        let leaf = Self::leaf_of(&leaves, ordinal)
            .ok_or_else(|| Error::IllegalState(format!("no leaf for ordinal {ordinal}")))?;
        let ret = Self::read_path(&leaf, ordinal)?;
        lock(&self.category_cache).put(ordinal, ret.clone());
        Ok(ret)
    }

    fn get_bulk_path(&self, ordinals: &[i32]) -> Result<Vec<FacetLabel>> {
        self.ensure_open()?;
        self.check_ordinal_bounds(ordinals)?;
        let mut bulk: Vec<Option<FacetLabel>> = {
            let mut cache = lock(&self.category_cache);
            // Every label cached (the common case): the answer as is. At
            // the first miss the hits so far carry over and the lookups go
            // on from there, so each ordinal is looked up (and touched in
            // the LRU order) once, as in Java.
            let mut hits = Vec::with_capacity(ordinals.len());
            for o in ordinals {
                match cache.get(o) {
                    Some(label) => hits.push(label.clone()),
                    None => break,
                }
            }
            if hits.len() == ordinals.len() {
                return Ok(hits);
            }
            let rest = ordinals.get(hits.len()..).unwrap_or_default();
            hits.into_iter()
                .map(Some)
                .chain(rest.iter().map(|o| cache.get(o).cloned()))
                .collect()
        };
        let mut uncached: Vec<(i32, usize)> = bulk
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_none())
            .map(|(i, _)| (ordinals[i], i))
            .collect();
        if !uncached.is_empty() {
            // Java's `InPlaceMergeSorter`: stable, by ordinal.
            uncached.sort_by_key(|&(ord, _)| ord);
            let leaves = self.index_reader.leaves();
            for &(ord, at) in &uncached {
                let leaf = Self::leaf_of(&leaves, ord)
                    .ok_or_else(|| Error::IllegalState(format!("no leaf for ordinal {ord}")))?;
                bulk[at] = Some(Self::read_path(&leaf, ord)?);
            }
            let mut cache = lock(&self.category_cache);
            for &(ord, at) in &uncached {
                if let Some(label) = &bulk[at] {
                    cache.put(ord, label.clone());
                }
            }
        }
        Ok(bulk.into_iter().flatten().collect())
    }

    fn get_size(&self) -> Result<i32> {
        self.ensure_open()?;
        Ok(self.index_reader.num_docs())
    }
}

#[cfg(test)]
mod tests;
