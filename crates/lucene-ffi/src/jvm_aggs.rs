//! The aggregation-tree entry points over [`lucene_engine::agg_tree`], which
//! holds the tree blob layout, the decoders and the evaluation.

use lucene_aggs::bucket_aggs;

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader;

fn handle(handle: u64) -> Result<std::sync::Arc<crate::registry::JvmReaderHandle>, FfiStatus> {
    jvm_reader::lookup(
        handle,
        "ffi_jvm_reader_aggregate_tree: unknown or already-closed handle",
    )
}

/// [`lucene_engine::agg_tree::aggregate_tree_blobs`] over `handle`'s reader.
pub(crate) fn aggregate_tree_blobs(
    h: u64,
    query_blob: &[u8],
    tree_blob: &[u8],
) -> Result<Vec<u8>, FfiStatus> {
    lucene_engine::agg_tree::aggregate_tree_blobs(&*handle(h)?, query_blob, tree_blob)
}

/// [`lucene_engine::agg_tree::aggregate_tree_blobs_cached`] over `handle`'s reader.
pub(crate) fn aggregate_tree_blobs_cached(
    h: u64,
    query_blob: &[u8],
    tree_blob: &[u8],
    cached: bool,
) -> Result<Vec<u8>, FfiStatus> {
    lucene_engine::agg_tree::aggregate_tree_blobs_cached(
        &*handle(h)?,
        query_blob,
        tree_blob,
        cached,
    )
}

/// The aggregations of a tree blob over the live matches of the query blob:
/// the encoded results (see the module doc) into `out`, their length into
/// `out_len` either way -- a too-small `out` is
/// [`FfiStatus::BufferTooSmall`], so a caller can size a second call.
///
/// # Safety
/// `query`/`tree` must be valid for `query_len`/`tree_len` bytes, `out` for
/// `cap` bytes (null when 0), `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_aggregate_tree(
    handle: u64,
    query: *const u8,
    query_len: usize,
    tree: *const u8,
    tree_len: usize,
    out: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    crate::error::guard(|| {
        if out_len.is_null() || (out.is_null() && cap > 0) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let q = unsafe { crate::raw::bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let t = unsafe { crate::raw::bytes_from_raw(tree, tree_len)? };
        let encoded = aggregate_tree_blobs(handle, q, t)?;
        // SAFETY: `out_len` is non-null and writable (caller contract).
        unsafe { *out_len = encoded.len() };
        if encoded.len() > cap {
            set_last_error(format!(
                "aggregation result: {} bytes, buffer holds {cap}",
                encoded.len()
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        if !encoded.is_empty() {
            // SAFETY: `out` is valid for `cap >= encoded.len()` bytes.
            unsafe { std::ptr::copy_nonoverlapping(encoded.as_ptr(), out, encoded.len()) };
        }
        Ok(())
    })
}

/// Sets the bytes the aggregation segment cache keeps parts in, across
/// every index of the process (0 turns it off and drops every part); the
/// parts past a smaller budget are dropped now.
#[no_mangle]
pub extern "C" fn ffi_agg_segment_cache_set_budget(budget_bytes: u64) -> i32 {
    crate::error::guard(|| {
        bucket_aggs::segment_cache::AggSegmentCache::global().set_budget(budget_bytes);
        if budget_bytes == 0 {
            bucket_aggs::segment_cache::AggSegmentCache::global().clear();
        }
        Ok(())
    })
}

/// The aggregation segment cache's counters into `out` (at least
/// [`AGG_CACHE_STATS`] `u64`s): hits, misses, evictions, entries, bytes,
/// budget.
///
/// # Safety
/// `out` must be valid for `len` `u64` writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_agg_segment_cache_stats(out: *mut u64, len: usize) -> i32 {
    crate::error::guard(|| {
        if out.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        if len < AGG_CACHE_STATS {
            set_last_error(format!("{len} slots for {AGG_CACHE_STATS} counters"));
            return Err(FfiStatus::BufferTooSmall);
        }
        let s = bucket_aggs::segment_cache::AggSegmentCache::global().stats();
        let values = [s.hits, s.misses, s.evictions, s.entries, s.bytes, s.budget];
        // SAFETY: `out` is valid for `len >= AGG_CACHE_STATS` writes.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()) };
        Ok(())
    })
}

/// The counters [`ffi_agg_segment_cache_stats`] writes.
pub const AGG_CACHE_STATS: usize = 6;

#[cfg(test)]
pub(crate) mod tests;
