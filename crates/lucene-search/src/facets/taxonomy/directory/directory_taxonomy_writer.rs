//! `DirectoryTaxonomyWriter`: a [`TaxonomyWriter`] keeping the taxonomy in
//! a Lucene index, one document per category, the document id being the
//! category's ordinal.
//!
//! As in Java, the writer runs its own [`IndexWriter`] (a
//! `LogByteSizeMergePolicy`, which only merges adjacent segments and so keeps
//! every document id; compound files, Java's default), answers lookups from
//! a [`TaxonomyWriterCache`] first and from a near-real-time reader over its
//! own index second, and fills the cache from that reader once enough
//! lookups have missed it (`cacheMissesUntilFill`). The same sequence of
//! calls produces the same ordinals and the same reader dump as Java's
//! (the fixtures compare those, not the segment layout).
//!
//! Every method takes `&self`; the writer's state sits behind one mutex,
//! which is Java's `synchronized` (Java's unsynchronised fast path in
//! `addCategory`, a cache hit, runs under the lock here).
//!
//! Differs from Java:
//! - [`DirectoryTaxonomyWriter::commit`] and
//!   [`DirectoryTaxonomyWriter::prepare_commit`] return the commit's
//!   generation (this port's `IndexWriter` hands out no commit sequence
//!   number); `prepare_commit` returns the generation the commit will get.
//! - `close` commits and then drops the index writer, where Java's
//!   `IndexWriter.close` commits a second time when anything changed in
//!   between (nothing can).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use lucene_index::deletion_policy::OpenMode;
use lucene_index::document::{
    BinaryDocValuesField, Document, NumericDocValuesField, Store, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::log::LogByteSizeMergePolicy;
use lucene_store::Directory;

use super::consts::{FIELD_PARENT_ORDINAL_NDV, FULL};
use super::taxonomy_index_arrays::TaxonomyIndexArrays;
use crate::directory_reader::{self, DirectoryReader};
use crate::facets::taxonomy::parallel_taxonomy_arrays::ParallelTaxonomyArrays;
use crate::facets::taxonomy::taxonomy_reader::{INVALID_ORDINAL, ROOT_ORDINAL};
use crate::facets::taxonomy::writercache::{LruTaxonomyWriterCache, TaxonomyWriterCache};
use crate::facets::taxonomy::{FacetLabel, TaxonomyWriter};
use crate::reader::{IndexReader, PostingsFlags};
use crate::{Error, Result};

/// `DirectoryTaxonomyWriter.INDEX_EPOCH`: the commit user data key holding
/// the taxonomy's epoch, in hexadecimal.
pub const INDEX_EPOCH: &str = "index.epoch";

/// `DEFAULT_CACHE_SIZE`.
const DEFAULT_CACHE_SIZE: i32 = 4000;

/// `defaultTaxonomyWriterCache()`: an [`LruTaxonomyWriterCache`] of 4000.
pub fn default_taxonomy_writer_cache() -> Box<dyn TaxonomyWriterCache> {
    Box::new(LruTaxonomyWriterCache::new(DEFAULT_CACHE_SIZE))
}

/// An index writer error as this crate's.
pub(crate) fn iw_error(e: lucene_index::index_writer::Error) -> Error {
    Error::DirectoryReader(directory_reader::Error::IndexWriter(e))
}

/// `Long.toString(v, 16)`.
fn java_long_hex(v: i64) -> String {
    if v < 0 {
        format!("-{:x}", v.unsigned_abs())
    } else {
        format!("{v:x}")
    }
}

/// `Long.parseLong(s, 16)`.
fn parse_java_long_hex(s: &str) -> Result<i64> {
    i64::from_str_radix(s, 16)
        .map_err(|_| Error::IllegalArgument(format!("For input string: \"{s}\" under radix 16")))
}

/// `new HashMap<>(data)` with [`INDEX_EPOCH`] put last, in Java's iteration
/// order (what `SegmentInfos` writes).
fn combined_commit_data(data: &[(String, String)], epoch: i64) -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::with_capacity(data.len().saturating_add(1));
    let mut put = |k: &str, v: &str| match entries.iter_mut().find(|(ek, _)| ek == k) {
        Some(e) => e.1 = v.to_string(),
        None => entries.push((k.to_string(), v.to_string())),
    };
    for (k, v) in data {
        put(k, v);
    }
    put(INDEX_EPOCH, &java_long_hex(epoch));
    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    let order = crate::mlt::java_hash_order(&keys);
    order.into_iter().map(|i| entries[i].clone()).collect()
}

/// `DirectoryReader.indexExists(dir)`.
fn index_exists(dir: &dyn Directory) -> Result<bool> {
    let files = dir.list_all()?;
    Ok(lucene_store::directory::last_commit_generation(&files)? != -1)
}

/// `DirectoryTaxonomyWriter.OrdinalMap`: where
/// [`DirectoryTaxonomyWriter::add_taxonomy`] records each added category's
/// old ordinal (in the source taxonomy) and new one (in this one).
pub trait OrdinalMap {
    /// `setSize(size)`: the source taxonomy's size, before any mapping.
    fn set_size(&mut self, size: i32) -> Result<()>;
    /// `addMapping(origOrdinal, newOrdinal)`.
    fn add_mapping(&mut self, orig_ordinal: i32, new_ordinal: i32) -> Result<()>;
    /// `addDone()`: no more mappings.
    fn add_done(&mut self) -> Result<()>;
    /// `getMap()`: `map[origOrdinal] = newOrdinal`.
    fn get_map(&mut self) -> Result<&[i32]>;
}

/// `Index X out of bounds for length N`.
fn out_of_bounds(index: i32, length: usize) -> Error {
    Error::IllegalArgument(format!("Index {index} out of bounds for length {length}"))
}

/// `MemoryOrdinalMap`: the map kept in memory.
#[derive(Debug, Default, Clone)]
pub struct MemoryOrdinalMap {
    map: Vec<i32>,
}

impl MemoryOrdinalMap {
    /// `new MemoryOrdinalMap()`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl OrdinalMap for MemoryOrdinalMap {
    fn set_size(&mut self, size: i32) -> Result<()> {
        let size = usize::try_from(size)
            .map_err(|_| Error::IllegalArgument(format!("Illegal Capacity: {size}")))?;
        self.map = vec![0; size];
        Ok(())
    }

    fn add_mapping(&mut self, orig_ordinal: i32, new_ordinal: i32) -> Result<()> {
        let len = self.map.len();
        let slot = usize::try_from(orig_ordinal)
            .ok()
            .and_then(|i| self.map.get_mut(i))
            .ok_or_else(|| out_of_bounds(orig_ordinal, len))?;
        *slot = new_ordinal;
        Ok(())
    }

    fn add_done(&mut self) -> Result<()> {
        Ok(())
    }

    fn get_map(&mut self) -> Result<&[i32]> {
        Ok(&self.map)
    }
}

/// `DiskOrdinalMap`: the mappings streamed to a file (big-endian ints: the
/// size, then `orig, new` pairs) and read back, the file deleted, on
/// [`OrdinalMap::get_map`].
#[derive(Debug)]
pub struct DiskOrdinalMap {
    map_file: PathBuf,
    map: Option<Vec<i32>>,
    out: Option<BufWriter<File>>,
}

impl DiskOrdinalMap {
    /// `new DiskOrdinalMap(mapFile)`: creates (truncates) the file.
    pub fn new(map_file: impl AsRef<Path>) -> Result<Self> {
        let map_file = map_file.as_ref().to_path_buf();
        let out = BufWriter::new(File::create(&map_file).map_err(io_error)?);
        Ok(DiskOrdinalMap {
            map_file,
            map: None,
            out: Some(out),
        })
    }

    fn write_int(&mut self, v: i32) -> Result<()> {
        match &mut self.out {
            Some(out) => out.write_all(&v.to_be_bytes()).map_err(io_error),
            None => Err(io_error(std::io::Error::other("Stream closed"))),
        }
    }
}

fn io_error(e: std::io::Error) -> Error {
    Error::Store(lucene_store::Error::Io(e))
}

fn read_int(input: &mut impl Read) -> Result<i32> {
    let mut b = [0u8; 4];
    input.read_exact(&mut b).map_err(io_error)?;
    Ok(i32::from_be_bytes(b))
}

impl OrdinalMap for DiskOrdinalMap {
    fn set_size(&mut self, size: i32) -> Result<()> {
        self.write_int(size)
    }

    fn add_mapping(&mut self, orig_ordinal: i32, new_ordinal: i32) -> Result<()> {
        self.write_int(orig_ordinal)?;
        self.write_int(new_ordinal)
    }

    fn add_done(&mut self) -> Result<()> {
        if let Some(mut out) = self.out.take() {
            out.flush().map_err(io_error)?;
        }
        Ok(())
    }

    fn get_map(&mut self) -> Result<&[i32]> {
        if self.map.is_none() {
            self.add_done()?;
            let mut input = BufReader::new(File::open(&self.map_file).map_err(io_error)?);
            let size = read_int(&mut input)?;
            let size = usize::try_from(size).map_err(|_| {
                Error::IllegalArgument(format!("-{} is a negative array size", size.unsigned_abs()))
            })?;
            let mut map = vec![0; size];
            for _ in 0..size {
                let orig = read_int(&mut input)?;
                let new = read_int(&mut input)?;
                let slot = usize::try_from(orig)
                    .ok()
                    .and_then(|i| map.get_mut(i))
                    .ok_or_else(|| out_of_bounds(orig, size))?;
                *slot = new;
            }
            drop(input);
            std::fs::remove_file(&self.map_file).map_err(io_error)?;
            self.map = Some(map);
        }
        Ok(self.map.as_deref().unwrap_or_default())
    }
}

/// The writer's state -- every field Java's methods read under `this`.
struct Inner<'d> {
    /// `None` once closed or rolled back.
    index_writer: Option<IndexWriter<'d>>,
    cache: Box<dyn TaxonomyWriterCache>,
    cache_misses: i32,
    next_id: i32,
    index_epoch: i64,
    cache_misses_until_fill: i32,
    should_fill_cache: bool,
    /// `readerManager`'s current reader; `Some` is `initializedReaderManager`.
    reader_manager: Option<DirectoryReader>,
    should_refresh_reader_manager: bool,
    cache_is_complete: bool,
    is_closed: bool,
    taxo_arrays: Option<TaxonomyIndexArrays>,
    /// A [`DirectoryTaxonomyWriter::prepare_commit`] awaits its commit.
    prepared: bool,
}

/// `DirectoryTaxonomyWriter`.
pub struct DirectoryTaxonomyWriter<'d> {
    dir: &'d dyn Directory,
    inner: Mutex<Inner<'d>>,
}

impl std::fmt::Debug for DirectoryTaxonomyWriter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectoryTaxonomyWriter")
            .finish_non_exhaustive()
    }
}

impl<'d> DirectoryTaxonomyWriter<'d> {
    /// `new DirectoryTaxonomyWriter(directory)`: [`OpenMode::CreateOrAppend`]
    /// and the default cache.
    pub fn open(dir: &'d dyn Directory) -> Result<Self> {
        Self::with_mode(dir, OpenMode::CreateOrAppend)
    }

    /// `new DirectoryTaxonomyWriter(directory, openMode)`.
    pub fn with_mode(dir: &'d dyn Directory, open_mode: OpenMode) -> Result<Self> {
        Self::with_cache(dir, open_mode, default_taxonomy_writer_cache())
    }

    /// `new DirectoryTaxonomyWriter(directory, openMode, cache)`.
    pub fn with_cache(
        dir: &'d dyn Directory,
        open_mode: OpenMode,
        cache: Box<dyn TaxonomyWriterCache>,
    ) -> Result<Self> {
        let existed = index_exists(dir)?;
        if open_mode == OpenMode::Append && !existed {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "no segments* file found in directory: files: {:?}",
                dir.list_all()?
            ))));
        }
        let mut index_writer = IndexWriter::open(
            dir,
            Vec::new(),
            "Lucene104",
            lucene_index::merge_policy::upgrade::LATEST,
        )
        .map_err(iw_error)?;
        index_writer.set_use_compound_file(true);
        index_writer.set_pluggable_merge_policy(Some(Arc::new(LogByteSizeMergePolicy::new())));

        let mut index_epoch = if existed {
            let infos = lucene_index::segment_infos::read_latest(dir).map_err(iw_or_store)?;
            match infos.user_data.iter().find(|(k, _)| k == INDEX_EPOCH) {
                Some((_, v)) => parse_java_long_hex(v)?,
                None => 1,
            }
        } else {
            1
        };
        let create = open_mode == OpenMode::Create;
        if create {
            index_epoch = index_epoch.saturating_add(1);
            if existed {
                // `OpenMode.CREATE` over an index: no segments, no user data.
                index_writer.delete_all().map_err(iw_error)?;
                index_writer.set_live_commit_data(Vec::new());
            }
        }
        let next_id = if create {
            0
        } else {
            i32::try_from(index_writer.committed_doc_count().map_err(iw_error)?)
                .map_err(|_| Error::IllegalState("taxonomy has more than 2^31 categories".into()))?
        };
        let writer = DirectoryTaxonomyWriter {
            dir,
            inner: Mutex::new(Inner {
                index_writer: Some(index_writer),
                cache,
                cache_misses: 0,
                next_id,
                index_epoch,
                cache_misses_until_fill: 11,
                should_fill_cache: true,
                reader_manager: None,
                should_refresh_reader_manager: false,
                cache_is_complete: next_id == 0,
                is_closed: false,
                taxo_arrays: None,
                prepared: false,
            }),
        };
        if next_id == 0 {
            writer.add_category(&FacetLabel::root())?;
        }
        Ok(writer)
    }

    fn lock(&self) -> MutexGuard<'_, Inner<'d>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `getDirectory()`.
    pub fn directory(&self) -> &'d dyn Directory {
        self.dir
    }

    /// `getTaxonomyEpoch()`: bumped by every `CREATE` open,
    /// [`Self::replace_taxonomy`] and [`Self::delete_all`].
    pub fn taxonomy_epoch(&self) -> i64 {
        self.lock().index_epoch
    }

    /// `setCacheMissesUntilFill(i)`.
    pub fn set_cache_misses_until_fill(&self, i: i32) -> Result<()> {
        let mut inner = self.lock();
        inner.ensure_open()?;
        inner.cache_misses_until_fill = i;
        Ok(())
    }

    /// `getCache()`: runs `f` over the writer's cache.
    pub fn with_cache_ref<R>(&self, f: impl FnOnce(&dyn TaxonomyWriterCache) -> R) -> R {
        f(self.lock().cache.as_ref())
    }

    /// `addTaxonomy(taxoDir, map)`: adds every category of the taxonomy in
    /// `taxo_dir`, recording in `map` each one's ordinal there and here.
    pub fn add_taxonomy(&self, taxo_dir: &dyn Directory, map: &mut dyn OrdinalMap) -> Result<()> {
        self.lock().ensure_open()?;
        let r = DirectoryReader::open(taxo_dir)?;
        map.set_size(r.num_docs())?;
        let mut base: i32 = 0;
        for leaf in r.leaves() {
            if let Some(terms) = leaf.reader.terms(FULL)? {
                let mut te = terms.iterator()?;
                while let Some(term) = te.next()? {
                    let cp = FacetLabel::from_term(term)?;
                    let ordinal = self.add_category(&cp)?;
                    let mut docs = te.postings(PostingsFlags::None)?;
                    map.add_mapping(docs.next_doc()?.wrapping_add(base), ordinal)?;
                }
            }
            base = base.wrapping_add(leaf.reader.max_doc());
        }
        map.add_done()
    }

    /// `replaceTaxonomy(taxoDir)`: this taxonomy becomes a copy of the one
    /// in `taxo_dir` (a new epoch, so readers reopen from scratch).
    pub fn replace_taxonomy(&self, taxo_dir: &dyn Directory) -> Result<()> {
        let mut inner = self.lock();
        let w = inner.writer_mut()?;
        w.delete_all().map_err(iw_error)?;
        w.add_indexes(&[taxo_dir]).map_err(iw_error)?;
        inner.reset_after_delete(self.dir, None)
    }

    /// `deleteAll()` (package-private in Java, for tests): every category gone, the
    /// root included, and a new epoch.
    pub fn delete_all(&self) -> Result<()> {
        let mut inner = self.lock();
        inner.writer_mut()?.delete_all().map_err(iw_error)?;
        inner.reset_after_delete(self.dir, Some(0))
    }

    fn do_commit(&self, prepare_only: bool) -> Result<i64> {
        let mut inner = self.lock();
        inner.ensure_open()?;
        let epoch = inner.index_epoch;
        let w = inner.writer_mut()?;
        let current = w
            .live_commit_data()
            .iter()
            .rev()
            .find(|(k, _)| k == INDEX_EPOCH)
            .map(|(_, v)| v.clone());
        let stale = match current {
            None => true,
            Some(v) => parse_java_long_hex(&v)? != epoch,
        };
        if stale {
            let data = combined_commit_data(w.live_commit_data(), epoch);
            w.set_live_commit_data(data);
        }
        if prepare_only {
            if inner.prepared {
                return Err(Error::IllegalState(
                    "prepareCommit was already called with no corresponding call to commit"
                        .to_string(),
                ));
            }
            let w = inner.writer_mut()?;
            w.prepare_commit().map_err(iw_error)?;
            let generation = w.segment_infos().generation.saturating_add(1);
            inner.prepared = true;
            return Ok(generation);
        }
        let prepared = std::mem::replace(&mut inner.prepared, false);
        let w = inner.writer_mut()?;
        let infos = if prepared {
            w.finish_commit().map_err(iw_error)?
        } else {
            w.commit().map_err(iw_error)?
        };
        Ok(infos.generation)
    }
}

/// What a near-real-time [`super::DirectoryTaxonomyReader`] needs of its
/// writer -- a trait so that the reader's one lifetime need not name the
/// writer's directory lifetime too.
pub(crate) trait TaxoNrtSource {
    /// `getTaxonomyEpoch()`.
    fn taxonomy_epoch(&self) -> i64;
    /// `DirectoryReader.open(getInternalIndexWriter())`: a near-real-time
    /// reader over everything added so far.
    fn open_nrt_reader(&self) -> Result<DirectoryReader>;
    /// `DirectoryReader.openIfChanged(reader)` for a reader from
    /// [`TaxoNrtSource::open_nrt_reader`].
    fn reopen_nrt_reader(&self, old: &DirectoryReader) -> Result<Option<DirectoryReader>>;
}

impl TaxoNrtSource for DirectoryTaxonomyWriter<'_> {
    fn taxonomy_epoch(&self) -> i64 {
        DirectoryTaxonomyWriter::taxonomy_epoch(self)
    }

    fn open_nrt_reader(&self) -> Result<DirectoryReader> {
        let mut inner = self.lock();
        let w = inner.writer_mut()?;
        open_nrt(w, self.dir, true)
    }

    fn reopen_nrt_reader(&self, old: &DirectoryReader) -> Result<Option<DirectoryReader>> {
        let mut inner = self.lock();
        let w = inner.writer_mut()?;
        refresh_nrt(w, self.dir, old, true)
    }
}

/// An error reading `segments_N`.
fn iw_or_store(e: lucene_index::segment_infos::Error) -> Error {
    Error::DirectoryReader(directory_reader::Error::SegmentInfos(e))
}

/// `new ReaderManager(indexWriter, applyAllDeletes, false)`'s first reader.
fn open_nrt(
    w: &mut IndexWriter<'_>,
    dir: &dyn Directory,
    apply_all_deletes: bool,
) -> Result<DirectoryReader> {
    let snapshot = w.nrt_snapshot(apply_all_deletes, false).map_err(iw_error)?;
    let mut reader = DirectoryReader::open_at(dir, snapshot.segment_infos)?;
    reader.nrt_hold = Some(snapshot.hold);
    Ok(reader)
}

/// `DirectoryReader.openIfChanged(reader)` for a near-real-time reader.
fn refresh_nrt(
    w: &mut IndexWriter<'_>,
    dir: &dyn Directory,
    old: &DirectoryReader,
    apply_all_deletes: bool,
) -> Result<Option<DirectoryReader>> {
    if w.nrt_is_current(&old.segment_infos) {
        return Ok(None);
    }
    let snapshot = w.nrt_snapshot(apply_all_deletes, false).map_err(iw_error)?;
    let mut reader = old.reopen_at(dir, snapshot.segment_infos)?;
    reader.nrt_hold = Some(snapshot.hold);
    Ok(Some(reader))
}

impl<'d> Inner<'d> {
    fn ensure_open(&self) -> Result<()> {
        if self.is_closed {
            return Err(Error::AlreadyClosed(
                "The taxonomy writer has already been closed".to_string(),
            ));
        }
        Ok(())
    }

    fn writer_mut(&mut self) -> Result<&mut IndexWriter<'d>> {
        self.index_writer
            .as_mut()
            .ok_or_else(|| Error::AlreadyClosed("this IndexWriter is closed".to_string()))
    }

    /// `initReaderManager()`.
    fn init_reader_manager(&mut self, dir: &dyn Directory) -> Result<()> {
        if self.reader_manager.is_none() {
            self.ensure_open()?;
            let reader = open_nrt(self.writer_mut()?, dir, false)?;
            self.reader_manager = Some(reader);
            self.should_refresh_reader_manager = false;
        }
        Ok(())
    }

    /// `refreshReaderManager()`.
    fn refresh_reader_manager(&mut self, dir: &dyn Directory) -> Result<()> {
        if self.should_refresh_reader_manager {
            if let Some(old) = self.reader_manager.take() {
                let refreshed = match self.index_writer.as_mut() {
                    Some(w) => refresh_nrt(w, dir, &old, false),
                    None => Ok(None),
                };
                self.reader_manager = Some(match refreshed {
                    Ok(Some(new)) => new,
                    Ok(None) => old,
                    Err(e) => {
                        self.reader_manager = Some(old);
                        return Err(e);
                    }
                });
                self.should_refresh_reader_manager = false;
            }
        }
        Ok(())
    }

    /// The tail of `replaceTaxonomy`/`deleteAll`, after the index writer's
    /// part. `next_id` is `None` for "the writer's document count".
    fn reset_after_delete(&mut self, dir: &dyn Directory, next_id: Option<i32>) -> Result<()> {
        self.should_refresh_reader_manager = true;
        self.init_reader_manager(dir)?;
        self.refresh_reader_manager(dir)?;
        self.next_id = match next_id {
            Some(id) => id,
            None => self.reader_manager.as_ref().map_or(0, |r| r.max_doc()),
        };
        self.taxo_arrays = None;
        self.cache.clear();
        self.cache_is_complete = false;
        self.should_fill_cache = true;
        self.cache_misses = 0;
        self.index_epoch = self.index_epoch.saturating_add(1);
        Ok(())
    }

    /// `findCategory(categoryPath)`.
    fn find_category(&mut self, dir: &dyn Directory, category_path: &FacetLabel) -> Result<i32> {
        let res = self.cache.get(category_path);
        if res >= 0 || self.cache_is_complete {
            return Ok(res);
        }
        self.cache_misses = self.cache_misses.saturating_add(1);
        self.perhaps_fill_cache(dir)?;
        let res = self.cache.get(category_path);
        if res >= 0 || self.cache_is_complete {
            return Ok(res);
        }
        self.init_reader_manager(dir)?;
        let term = category_path.to_term();
        let mut doc = -1;
        if let Some(reader) = &self.reader_manager {
            for leaf in reader.leaves() {
                let Some(terms) = leaf.reader.terms(FULL)? else {
                    continue;
                };
                let mut te = terms.iterator()?;
                if te.try_seek_exact(term.as_bytes())? {
                    let mut docs = te.postings(PostingsFlags::None)?;
                    doc = docs.next_doc()?.wrapping_add(leaf.doc_base);
                    break;
                }
            }
        }
        if doc > 0 {
            self.add_to_cache(dir, category_path, doc)?;
        }
        Ok(doc)
    }

    /// `internalAddCategory(cp)`: adds `cp` after every missing ancestor.
    /// A loop over the missing ancestors, where Java recurses.
    fn internal_add_category(&mut self, dir: &dyn Directory, cp: &FacetLabel) -> Result<i32> {
        // The missing ancestors, deepest first; `parent` is the ordinal of
        // the nearest existing one.
        let mut missing = vec![cp.clone()];
        let mut parent;
        loop {
            let Some(last) = missing.last() else {
                return Err(Error::IllegalState("no category to add".into()));
            };
            let length = last.len();
            if length > 1 {
                let parent_path = last.subpath(length.saturating_sub(1));
                parent = self.find_category(dir, &parent_path)?;
                if parent < 0 {
                    missing.push(parent_path);
                    continue;
                }
            } else if length == 1 {
                parent = ROOT_ORDINAL;
            } else {
                parent = INVALID_ORDINAL;
            }
            break;
        }
        while let Some(path) = missing.pop() {
            parent = self.add_category_document(dir, &path, parent)?;
        }
        Ok(parent)
    }

    /// `addCategoryDocument(categoryPath, parent)`.
    fn add_category_document(
        &mut self,
        dir: &dyn Directory,
        category_path: &FacetLabel,
        parent: i32,
    ) -> Result<i32> {
        let field_path = category_path.to_term();
        let mut d = Document::new();
        d.add(NumericDocValuesField::new(
            FIELD_PARENT_ORDINAL_NDV,
            i64::from(parent),
        ));
        d.add(BinaryDocValuesField::new(FULL, field_path.as_bytes()));
        d.add(StringField::new(FULL, field_path, Store::No));
        self.writer_mut()?
            .add_fields_document(&d)
            .map_err(iw_error)?;
        let id = self.next_id;
        self.next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::IllegalState("taxonomy has more than 2^31 categories".into()))?;
        self.should_refresh_reader_manager = true;
        self.taxo_arrays(dir)?.add(id, parent);
        self.add_to_cache(dir, category_path, id)?;
        Ok(id)
    }

    /// `addToCache(categoryPath, id)`.
    fn add_to_cache(
        &mut self,
        dir: &dyn Directory,
        category_path: &FacetLabel,
        id: i32,
    ) -> Result<()> {
        if self.cache.put(category_path, id) {
            self.refresh_reader_manager(dir)?;
            self.cache_is_complete = false;
        }
        Ok(())
    }

    /// `perhapsFillCache()`.
    fn perhaps_fill_cache(&mut self, dir: &dyn Directory) -> Result<()> {
        if self.cache_misses < self.cache_misses_until_fill || !self.should_fill_cache {
            return Ok(());
        }
        self.should_fill_cache = false;
        self.init_reader_manager(dir)?;
        let mut aborted = false;
        if let Some(reader) = &self.reader_manager {
            let cache = &mut self.cache;
            'leaves: for leaf in reader.leaves() {
                let Some(terms) = leaf.reader.terms(FULL)? else {
                    continue;
                };
                let mut te = terms.iterator()?;
                while let Some(term) = te.next()? {
                    if cache.is_full() {
                        aborted = true;
                        break 'leaves;
                    }
                    let cp = FacetLabel::from_term(term)?;
                    let mut docs = te.postings(PostingsFlags::None)?;
                    let doc = docs.next_doc()?.wrapping_add(leaf.doc_base);
                    // Java asserts nothing is evicted: the cache was not full.
                    cache.put(&cp, doc);
                }
            }
        }
        self.cache_is_complete = !aborted;
        if self.cache_is_complete {
            self.reader_manager = None;
        }
        Ok(())
    }

    /// `getTaxoArrays()`.
    fn taxo_arrays(&mut self, dir: &dyn Directory) -> Result<&mut TaxonomyIndexArrays> {
        if self.taxo_arrays.is_none() {
            self.init_reader_manager(dir)?;
            let arrays = match &self.reader_manager {
                Some(reader) => TaxonomyIndexArrays::new(reader)?,
                None => return Err(Error::IllegalState("no taxonomy reader".into())),
            };
            self.taxo_arrays = Some(arrays);
        }
        self.taxo_arrays
            .as_mut()
            .ok_or_else(|| Error::IllegalState("no taxonomy arrays".into()))
    }

    /// `doClose()` / `closeResources()`.
    fn do_close(&mut self) {
        self.is_closed = true;
        self.reader_manager = None;
        self.cache.close();
    }
}

impl TaxonomyWriter for DirectoryTaxonomyWriter<'_> {
    fn add_category(&self, category_path: &FacetLabel) -> Result<i32> {
        let mut inner = self.lock();
        inner.ensure_open()?;
        let res = inner.cache.get(category_path);
        if res >= 0 {
            return Ok(res);
        }
        let res = inner.find_category(self.dir, category_path)?;
        if res >= 0 {
            return Ok(res);
        }
        inner.internal_add_category(self.dir, category_path)
    }

    fn get_parent(&self, ordinal: i32) -> Result<i32> {
        let mut inner = self.lock();
        inner.ensure_open()?;
        let size = inner.next_id;
        if ordinal < 0 || ordinal >= size {
            return Err(out_of_bounds(ordinal, usize::try_from(size).unwrap_or(0)));
        }
        let arrays = inner.taxo_arrays(self.dir)?;
        Ok(arrays.parents().get(ordinal))
    }

    fn get_size(&self) -> Result<i32> {
        let inner = self.lock();
        inner.ensure_open()?;
        Ok(inner.next_id)
    }

    fn set_live_commit_data(&self, commit_user_data: Vec<(String, String)>) -> Result<()> {
        let mut inner = self.lock();
        let epoch = inner.index_epoch;
        let data = combined_commit_data(&commit_user_data, epoch);
        inner.writer_mut()?.set_live_commit_data(data);
        Ok(())
    }

    fn live_commit_data(&self) -> Result<Vec<(String, String)>> {
        let mut inner = self.lock();
        let epoch = inner.index_epoch;
        Ok(combined_commit_data(
            inner.writer_mut()?.live_commit_data(),
            epoch,
        ))
    }

    fn prepare_commit(&self) -> Result<i64> {
        self.do_commit(true)
    }

    fn commit(&self) -> Result<i64> {
        self.do_commit(false)
    }

    fn rollback(&self) -> Result<()> {
        let mut inner = self.lock();
        inner.ensure_open()?;
        if let Some(mut w) = inner.index_writer.take() {
            w.rollback();
        }
        inner.prepared = false;
        inner.do_close();
        Ok(())
    }

    fn close(&self) -> Result<()> {
        if self.lock().is_closed {
            return Ok(());
        }
        self.commit()?;
        let mut inner = self.lock();
        inner.index_writer = None;
        inner.do_close();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
