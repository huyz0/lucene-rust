//! Live search entry points for the plugin, over [`lucene_engine::live`]:
//! documents still in the engine writer's buffers, searched and fetched
//! without a refresh. The view machinery is in the engine crate; the
//! functions below decode the caller's pointers and call it.

use std::sync::atomic::Ordering;

use crate::engine_writer;
use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::bytes_from_raw;

pub(crate) use lucene_engine::live::*;

/// Lets searches of `writer`'s buffers read a view up to `millis`
/// milliseconds old: the writer's warming thread opens one every few
/// milliseconds, its chunks written, and the searches meanwhile share it,
/// its readers and their caches, writing nothing themselves. A search then
/// misses the writes of at most that long. 0 (the default) gives each search
/// a view of every write acknowledged before it.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_live_staleness(writer: u64, millis: i64) -> i32 {
    guard(|| {
        let w = engine_writer::lookup(writer)?;
        let ms = u64::try_from(millis).map_err(|_| {
            set_last_error("live staleness: negative");
            FfiStatus::InvalidArgument
        })?;
        if ms == 0 {
            w.live_warmer().stop_sharing();
        } else {
            w.live_warmer()
                .max_staleness_ms
                .store(ms, Ordering::Relaxed);
        }
        Ok(())
    })
}

/// Under a staleness bound, makes every later search read a view opened
/// from now on, so it holds every write acknowledged before this call:
/// what lets `refresh=wait_for` answer at once.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_live_refresh_shared(writer: u64) -> i32 {
    guard(|| {
        engine_writer::lookup(writer)?
            .live_warmer()
            .refresh_shared();
        Ok(())
    })
}

/// Opens a view of every buffered document a reader of the segment list at
/// version `reader_version` (a commit's or an NRT snapshot's) lacks (`-1`:
/// the documents in no published segment),
/// its handle into `*out_view` and its blob (see the module doc) as an owned
/// buffer.
///
/// # Safety
/// `out_view`, `out_ptr` and `out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_live_view_open(
    writer: u64,
    reader_version: i64,
    out_view: *mut u64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_view.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let w = engine_writer::lookup(writer)?;
        let commit = (reader_version >= 0).then_some(reader_version);
        let warmer = w.live_warmer();
        if warmer.max_staleness().is_some() {
            // The warmer opens the views searches share, and forgets the
            // shared one once no search comes.
            searched_lean(&w);
        }
        let h = match warmer.shared_view(commit) {
            Some(h) => h,
            None => {
                warmer.asked(commit);
                open_handle(&w, commit, false)?
            }
        };
        let blob = h.blob.clone();
        // A view searches share keeps its handle: each open of it counts.
        let id = *h
            .id
            .get_or_init(|| NEXT_VIEW.fetch_add(1, Ordering::Relaxed));
        let mut open = views()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = open.entry(id).or_insert((h, 0));
        entry.1 = entry.1.saturating_add(1);
        drop(open);
        // SAFETY: caller contract.
        unsafe {
            *out_view = id;
            crate::ffm_bridge::give(blob, out_ptr, out_len);
        }
        Ok(())
    })
}

/// Closes one open of a view: a view searches share has one handle for all
/// its opens, closed as many times. Its buffers may go once no other view
/// reads them.
#[no_mangle]
pub extern "C" fn ffi_live_view_close(view: u64) -> i32 {
    guard(|| {
        let mut open = views()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entry) = open.get_mut(&view) else {
            set_last_error("live view: unknown or already-closed handle");
            return Err(FfiStatus::InvalidHandle);
        };
        entry.1 = entry.1.saturating_sub(1);
        let last = (entry.1 == 0).then(|| open.remove(&view));
        drop(open);
        // The view closes when its last handle goes, outside the lock: see
        // `ViewHandle`'s drop.
        drop(last);
        Ok(())
    })
}

/// A reader handle over the reader at `handle` and `view`'s buffers as
/// segments after its own, into `*out_handle`: every search the JVM reader
/// takes ([`crate::jvm_reader`], aggregations, sorts) then reads the
/// buffered documents too, numbered as [`ffi_jvm_reader_search_live`]
/// numbers them. The buffers are written into memory as their flush would
/// write them, once per view; the deletes since the reader's commit, and
/// the buffers' own, are its deleted documents. The handle belongs to the
/// view: closing the view closes it, and the caller must not.
///
/// With a request (`query` as `decode_request` reads it, `tree` an
/// aggregation tree or empty, and `extra` more queries the search runs, a
/// rescore's, each an `i32` length then the blob, or empty), the buffers
/// are written without their postings and the handle holds only the terms
/// the request reads: it serves searches of that request alone. Without
/// one (`query_len` 0, when `extra` is not read), or for a request reading
/// what that cannot give, the handle holds every term.
///
/// # Safety
/// `out_handle` valid for one write; `query`/`tree`/`extra` readable for
/// their lengths when non-zero.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_reader(
    handle: u64,
    view_id: u64,
    query: *const u8,
    query_len: usize,
    tree: *const u8,
    tree_len: usize,
    extra: *const u8,
    extra_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        let _timer = lucene_index::live_timing::start(lucene_index::live_timing::Phase::ViewReader);
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let v = view(view_id)?;
        // SAFETY: caller contract.
        let extra = (extra_len > 0 && query_len > 0)
            .then(|| unsafe { bytes_from_raw(extra, extra_len) })
            .transpose()?;
        // SAFETY: caller contract.
        let query = (query_len > 0)
            .then(|| unsafe { bytes_from_raw(query, query_len) })
            .transpose()?;
        // SAFETY: caller contract.
        let tree = (tree_len > 0)
            .then(|| unsafe { bytes_from_raw(tree, tree_len) })
            .transpose()?;
        if query.is_some() && v.writer.live_warmer().max_staleness().is_some() {
            v.writer
                .live_warmer()
                .note_reader(handle, v.commit, query, tree, extra);
        }
        let id = view_reader_of(handle, &v, query, tree, extra)?;
        // SAFETY: caller contract.
        unsafe { *out_handle = id };
        Ok(())
    })
}

/// [`crate::jvm_knn::ffi_jvm_reader_knn`] over the reader at `handle` and
/// `view_id`'s buffers, numbered as [`ffi_jvm_reader_search_live`] numbers
/// them: the request, outputs and errors are that call's. The buffers are
/// read through a handle of [`ffi_live_view_reader`]'s that holds only the
/// filter's terms (every term for a nested field, whose parent filter it
/// does not know), cached with the view; their vectors are searched
/// exactly, as a realtime chunk keeps no graph.
///
/// # Safety
/// As [`crate::jvm_knn::ffi_jvm_reader_knn`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_live_view_knn(
    handle: u64,
    view_id: u64,
    request: *const u8,
    request_len: usize,
    out_docs: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
    out_count: *mut usize,
) -> i32 {
    guard(|| {
        if out_docs.is_null() || out_scores.is_null() || out_count.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(request, request_len)? };
        let hits = live_knn(handle, view_id, blob, buf_len)?;
        // SAFETY: caller contract.
        unsafe { crate::jvm_knn::write_hits(&hits, out_docs, out_scores, buf_len, out_count) }
    })
}

/// What a search Lucene answers reads the view's buffers through, into
/// `*out_ptr`/`*out_len`: the deletes since the commit of the reader at
/// `handle` (the view's base), and each leaf of the view as a segment with
/// its postings, for the JVM to open.
///
/// Little-endian: `segments: i32`, then per segment of the reader its
/// live documents as of the view (`words: i32`, `-1` when none were
/// deleted since the commit, then that many `i64`); then `leaves: i32`, and
/// per leaf `kind: u8` (`0` a segment in memory, `1` the segment its buffer
/// was flushed as, in the index's directory), the segment's name (`i32`
/// length, UTF-8), its 16-byte id, its codec's name (as the name), the leaf's first document in it
/// (`i32`), and with kind `0` its files (`count: i32`, each a name as above,
/// `len: i64` and the bytes). A leaf of a flushed buffer may be a part of
/// that segment, which also holds documents added after the view.
///
/// A buffer taken for a flush since the view opened is waited for up to
/// `wait_ms` to publish; [`FfiStatus::InvalidArgument`] when it has not.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_lucene(
    handle: u64,
    view_id: u64,
    wait_ms: i64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let out = view_lucene(handle, view_id, wait_ms)?;
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Stops keeping the published buffers the segment list at version
/// `reader_version` (a commit's or an NRT snapshot's) holds, once no open
/// view reads them.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_release_live(writer: u64, reader_version: i64) -> i32 {
    guard(|| {
        let w = engine_writer::lookup(writer)?;
        w.live_warmer().unshare_before(reader_version);
        w.writer().release_live(reader_version).map_err(live_error)
    })
}

/// The bounds of field `field`'s points in buffer `buffer` of `view` as the view
/// opened ([`lucene_index::concurrent_writer::ConcurrentIndexWriter::live_point_bounds`]),
/// as an owned buffer: the point count as a little-endian `u64`, then the
/// packed least values, then the packed greatest; empty when the buffer
/// holds no point of the field.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_point_bounds_alloc(
    view_id: u64,
    buffer: i32,
    field: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let out = live_point_bounds_blob(view_id, buffer, field)?;
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Document `doc` of buffer `buffer` of `view`, its stored fields encoded
/// as [`crate::jvm_fetch`]'s document blob, as an owned buffer.
///
/// # Safety
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_document_alloc(
    view_id: u64,
    buffer: i32,
    doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        let out = live_document_blob(view_id, buffer, doc)?;
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// Document `doc` of buffer `buffer` of `view`: its doc values of the
/// `count` fields numbered at `fields`, encoded as [`crate::jvm_fetch`]'s
/// doc-values blob, as an owned buffer. A buffer has no terms dictionary:
/// a document's terms are numbered `0..n` in order, ordinals of that
/// document alone.
///
/// # Safety
/// `fields` valid for `count` `i32` reads (null when 0); `out_ptr`/`out_len`
/// valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_live_view_doc_values_alloc(
    view_id: u64,
    buffer: i32,
    doc: i32,
    fields: *const i32,
    count: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::clear_out(out_ptr, out_len)? };
        if fields.is_null() && count > 0 {
            return Err(FfiStatus::NullPointer);
        }
        let fields: &[i32] = if count == 0 {
            &[]
        } else {
            // SAFETY: caller contract, non-null checked above.
            unsafe { std::slice::from_raw_parts(fields, count) }
        };
        let out = live_doc_values_blob(view_id, buffer, doc, fields)?;
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(out, out_ptr, out_len) };
        Ok(())
    })
}

/// [`crate::jvm_reader::ffi_jvm_reader_search`] over the reader at
/// `handle` and the buffers of `view` together: a buffered hit is
/// numbered after the reader's `maxDoc`, the buffers in the view's order.
/// Every field's average length is over both. `*out_supported` is false,
/// and nothing else is written, when the query reads something the buffers
/// cannot answer (see `live_request`) or carries a `min_score`; the
/// caller then makes the buffers part of a reader first.
///
/// # Safety
/// As [`crate::jvm_reader::ffi_jvm_reader_search`]; `out_supported` valid
/// for one write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_search_live(
    handle: u64,
    view_id: u64,
    query: *const u8,
    query_len: usize,
    top_n: usize,
    count_limit: i64,
    out_docs: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
    out_hit_count: *mut usize,
    out_total: *mut i64,
    out_total_is_lower_bound: *mut bool,
    out_supported: *mut bool,
) -> i32 {
    guard(|| {
        let _timer =
            lucene_index::live_timing::start(lucene_index::live_timing::Phase::PlainSearch);
        if out_hit_count.is_null()
            || out_total.is_null()
            || out_total_is_lower_bound.is_null()
            || out_supported.is_null()
        {
            return Err(FfiStatus::NullPointer);
        }
        if buf_len < top_n {
            return Err(FfiStatus::BufferTooSmall);
        }
        if top_n > 0 && (out_docs.is_null() || out_scores.is_null()) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        let Some((hits, total, lower_bound)) =
            search_live(handle, view_id, blob, top_n, count_limit)?
        else {
            // SAFETY: caller contract.
            unsafe { *out_supported = false };
            return Ok(());
        };
        // SAFETY: caller contract; `hits.len() <= top_n <= buf_len`.
        unsafe {
            for (i, hit) in hits.iter().enumerate() {
                *out_docs.add(i) = hit.doc_id;
                *out_scores.add(i) = hit.score;
            }
            *out_hit_count = hits.len();
            *out_total = total;
            *out_total_is_lower_bound = lower_bound;
            *out_supported = true;
        }
        Ok(())
    })
}
