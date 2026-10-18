//! `TaxonomyIndexArrays`: the [`ParallelTaxonomyArrays`] read off a taxonomy
//! index -- `parents` from each document's `$parent_ndv$` doc value,
//! `children` and `siblings` derived from them on first use.
//!
//! The arrays are chunked (8192 ints a chunk), as in Java, so that a reader
//! reopened over a grown taxonomy shares every full chunk of the old one.
//! Java shares chunks by reference and reallocates the one chunk of
//! `children` a reopen would otherwise write into; here each chunk is an
//! `Arc`, and a write to a shared chunk copies it (`Arc::make_mut`), which is
//! the same sharing with the copy made in one place.

use std::sync::{Arc, OnceLock};

use super::consts::FIELD_PARENT_ORDINAL_NDV;
use crate::directory_reader::DirectoryReader;
use crate::facets::taxonomy::parallel_taxonomy_arrays::{
    IntArray, IntChunks, ParallelTaxonomyArrays,
};
use crate::facets::taxonomy::taxonomy_reader::INVALID_ORDINAL;
use crate::reader::IndexReader;
use crate::{Error, Result};

const CHUNK_SIZE_BITS: u32 = 13;
/// `CHUNK_SIZE`.
pub const CHUNK_SIZE: usize = 1 << CHUNK_SIZE_BITS;
const CHUNK_MASK: usize = CHUNK_SIZE - 1;

/// `RamUsageEstimator.NUM_BYTES_OBJECT_HEADER` (compressed oops).
const NUM_BYTES_OBJECT_HEADER: i64 = 12;
/// `RamUsageEstimator.NUM_BYTES_OBJECT_REF` (compressed oops).
const NUM_BYTES_OBJECT_REF: i64 = 4;
/// `RamUsageEstimator.shallowSizeOf(ChunkedIntArray)`: a header and one
/// reference, aligned to 8.
const SHALLOW_CHUNKED_INT_ARRAY: i64 = 16;
/// `ArrayUtil.MAX_ARRAY_LENGTH`.
const MAX_ARRAY_LENGTH: usize = (i32::MAX as usize) - 16;

/// `ArrayUtil.oversize(minTargetSize, Integer.BYTES)` on a 64-bit JVM.
pub(crate) fn oversize_ints(min_target_size: usize) -> usize {
    if min_target_size == 0 {
        return 0;
    }
    let extra = (min_target_size >> 3).max(3);
    let new_size = min_target_size.saturating_add(extra);
    if new_size.saturating_add(7) > MAX_ARRAY_LENGTH {
        return MAX_ARRAY_LENGTH;
    }
    new_size.saturating_add(1) & 0x7fff_fffe
}

/// `TaxonomyIndexArrays.ChunkedIntArray`.
#[derive(Debug, Clone)]
pub struct ChunkedIntArray {
    /// `Arc<[i32]>`, not `Arc<Vec<i32>>`: a read is one load for the chunk's
    /// pointer and one for the value, as Java's `values[i >> 13][i & mask]`
    /// -- the walks down `children`/`siblings` are a chain of these.
    values: Vec<Arc<[i32]>>,
}

impl ChunkedIntArray {
    /// `allocateChunkedArray(size, 0)`: zeros.
    fn zeroed(size: usize) -> Self {
        let chunks = (size >> CHUNK_SIZE_BITS).saturating_add(1);
        let mut values = Vec::with_capacity(chunks);
        for i in 0..chunks {
            let len = if i.saturating_add(1) == chunks {
                size & CHUNK_MASK
            } else {
                CHUNK_SIZE
            };
            values.push(Arc::from(vec![0; len]));
        }
        ChunkedIntArray { values }
    }

    /// `allocateChunkedArray(size, old.values.length - 1)` then
    /// `copyChunkedArray(old.values, new)`: `old`'s full chunks shared, its
    /// last chunk copied into a fresh one, zeros after.
    fn grown_from(old: &ChunkedIntArray, size: usize) -> Self {
        let chunks = (size >> CHUNK_SIZE_BITS).saturating_add(1);
        let old_chunks = old.values.len();
        let mut values = Vec::with_capacity(chunks);
        for i in 0..chunks {
            if i.saturating_add(1) < old_chunks {
                values.push(Arc::clone(&old.values[i]));
                continue;
            }
            let len = if i.saturating_add(1) == chunks {
                size & CHUNK_MASK
            } else {
                CHUNK_SIZE
            };
            let mut chunk = vec![0; len];
            if i.saturating_add(1) == old_chunks {
                let last = &old.values[i];
                let n = last.len().min(len);
                chunk[..n].copy_from_slice(&last[..n]);
            }
            values.push(Arc::from(chunk));
        }
        ChunkedIntArray { values }
    }

    fn len(&self) -> usize {
        match self.values.last() {
            Some(last) => {
                (self.values.len().saturating_sub(1) << CHUNK_SIZE_BITS).saturating_add(last.len())
            }
            None => 0,
        }
    }

    #[inline]
    fn at(&self, i: usize) -> i32 {
        self.values[i >> CHUNK_SIZE_BITS][i & CHUNK_MASK]
    }

    /// `set(i, val)`.
    fn set(&mut self, i: usize, val: i32) {
        // Copy on write: a chunk shared with another reader is copied first.
        Arc::make_mut(&mut self.values[i >> CHUNK_SIZE_BITS])[i & CHUNK_MASK] = val;
    }
}

impl IntArray for ChunkedIntArray {
    #[inline]
    fn get(&self, i: i32) -> i32 {
        let i = usize::try_from(i).expect("negative taxonomy array index");
        self.at(i)
    }

    fn length(&self) -> i32 {
        i32::try_from(self.len()).unwrap_or(i32::MAX)
    }

    fn chunks(&self) -> Option<IntChunks<'_>> {
        Some(IntChunks::new(CHUNK_SIZE_BITS, &self.values))
    }
}

/// `TaxonomyIndexArrays`.
#[derive(Debug)]
pub struct TaxonomyIndexArrays {
    parents: ChunkedIntArray,
    /// `children` and `siblings`, computed on first use
    /// (`initChildrenSiblings`).
    children_siblings: OnceLock<(ChunkedIntArray, ChunkedIntArray)>,
}

impl TaxonomyIndexArrays {
    /// `TaxonomyIndexArrays(int[][] parents)`.
    fn from_parents(parents: ChunkedIntArray) -> Self {
        TaxonomyIndexArrays {
            parents,
            children_siblings: OnceLock::new(),
        }
    }

    /// `new TaxonomyIndexArrays(reader)`.
    pub fn new(reader: &DirectoryReader) -> Result<Self> {
        let max_doc = usize::try_from(reader.max_doc()).unwrap_or(0);
        let mut parents = ChunkedIntArray::zeroed(max_doc);
        if max_doc > 0 {
            init_parents(&mut parents, reader, 0)?;
            parents.set(0, INVALID_ORDINAL);
        }
        Ok(Self::from_parents(parents))
    }

    /// `new TaxonomyIndexArrays(reader, copyFrom)`: `copy_from`'s parents
    /// (and children and siblings, when it has computed them) reused, only
    /// the ordinals added since read and computed.
    pub fn copy_from(reader: &DirectoryReader, copy_from: &TaxonomyIndexArrays) -> Result<Self> {
        let max_doc = usize::try_from(reader.max_doc()).unwrap_or(0);
        let mut parents = ChunkedIntArray::grown_from(&copy_from.parents, max_doc);
        init_parents(&mut parents, reader, copy_from.parents.len())?;
        let arrays = Self::from_parents(parents);
        if let Some((children, siblings)) = copy_from.children_siblings.get() {
            let len = arrays.parents.len();
            let mut children = ChunkedIntArray::grown_from(children, len);
            let mut siblings = ChunkedIntArray::grown_from(siblings, len);
            compute_children_siblings(
                &arrays.parents,
                &mut children,
                &mut siblings,
                copy_from.parents.len(),
            );
            let _ = arrays.children_siblings.set((children, siblings));
        }
        Ok(arrays)
    }

    /// `add(ordinal, parentOrdinal)`: records `ordinal`'s parent, growing
    /// the parents array (`ArrayUtil.oversize`) when it is too short. The
    /// writer's arrays never compute children, and a grown array starts
    /// without them, as Java's new instance does.
    pub(crate) fn add(&mut self, ordinal: i32, parent_ordinal: i32) {
        let Ok(at) = usize::try_from(ordinal) else {
            return;
        };
        if at >= self.parents.len() {
            let size = oversize_ints(at.saturating_add(1));
            self.parents = ChunkedIntArray::grown_from(&self.parents, size);
            self.children_siblings = OnceLock::new();
        }
        self.parents.set(at, parent_ordinal);
    }

    fn children_siblings(&self) -> &(ChunkedIntArray, ChunkedIntArray) {
        self.children_siblings.get_or_init(|| {
            let len = self.parents.len();
            let mut children = ChunkedIntArray::zeroed(len);
            let mut siblings = ChunkedIntArray::zeroed(len);
            compute_children_siblings(&self.parents, &mut children, &mut siblings, 0);
            (children, siblings)
        })
    }

    /// `ramBytesUsed()`: Java's shallow figure (the objects, not the ints).
    pub fn ram_bytes_used(&self) -> i64 {
        // constants only: `HEADER + 3 * REF + 1` cannot overflow
        const SHALLOW: i64 = NUM_BYTES_OBJECT_HEADER + 3 * NUM_BYTES_OBJECT_REF + 1;
        let mut ram = SHALLOW;
        ram = ram.saturating_add(SHALLOW_CHUNKED_INT_ARRAY);
        if self.children_siblings.get().is_some() {
            ram = ram.saturating_add(SHALLOW_CHUNKED_INT_ARRAY.saturating_mul(2));
        }
        ram
    }
}

impl ParallelTaxonomyArrays for TaxonomyIndexArrays {
    fn parents(&self) -> &dyn IntArray {
        &self.parents
    }

    fn children(&self) -> &dyn IntArray {
        &self.children_siblings().0
    }

    fn siblings(&self) -> &dyn IntArray {
        &self.children_siblings().1
    }
}

/// `computeChildrenSiblings(first)`.
fn compute_children_siblings(
    parents: &ChunkedIntArray,
    children: &mut ChunkedIntArray,
    siblings: &mut ChunkedIntArray,
    first: usize,
) {
    let length = parents.len();
    for i in first..length {
        children.set(i, INVALID_ORDINAL);
    }
    let mut first = first;
    if first == 0 {
        first = 1;
        if length > 0 {
            siblings.set(0, INVALID_ORDINAL);
        }
    }
    for i in first..length {
        // A parent outside the taxonomy (a corrupt index: Java fails with an
        // `ArrayIndexOutOfBoundsException`) leaves the category unlinked.
        match usize::try_from(parents.at(i)).ok().filter(|&p| p < length) {
            Some(p) => {
                siblings.set(i, children.at(p));
                children.set(p, i32::try_from(i).unwrap_or(i32::MAX));
            }
            None => siblings.set(i, INVALID_ORDINAL),
        }
    }
}

/// `initParents(parentsArray, reader, first)`.
fn init_parents(
    parents: &mut ChunkedIntArray,
    reader: &DirectoryReader,
    first: usize,
) -> Result<()> {
    if usize::try_from(reader.max_doc()).unwrap_or(0) == first {
        return Ok(());
    }
    for leaf in reader.leaves() {
        let leaf_docs = usize::try_from(leaf.reader.max_doc()).unwrap_or(0);
        let doc_base = usize::try_from(leaf.doc_base).unwrap_or(0);
        if doc_base.saturating_add(leaf_docs) <= first {
            continue;
        }
        let Some(mut values) = leaf.reader.numeric_doc_values(FIELD_PARENT_ORDINAL_NDV)? else {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "Parent data field {FIELD_PARENT_ORDINAL_NDV} does not exist"
            ))));
        };
        for doc in first.saturating_sub(doc_base)..leaf_docs {
            let target = i32::try_from(doc).unwrap_or(i32::MAX);
            if !values.advance_exact(target)? {
                return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                    "Missing parent data for category {}",
                    doc.saturating_add(doc_base)
                ))));
            }
            let parent = i32::try_from(values.long_value())
                .map_err(|_| Error::IllegalArgument("integer overflow".to_string()))?;
            parents.set(doc.saturating_add(doc_base), parent);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
