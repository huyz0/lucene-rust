//! The whole-reader KNN entry points over [`lucene_engine::knn`], which holds
//! the request blob layout and the search.

use lucene_engine::knn::{decode, recall_stats_json, search};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::jvm_reader::lookup;
use crate::raw::bytes_from_raw;

/// Every tuner's statistics ([`recall_stats_json`]), UTF-8 JSON handed
/// back as an owned buffer (`ffi_jvm_free_bytes`).
///
/// # Safety
/// `out_ptr` and `out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_knn_recall_stats(out_ptr: *mut *mut u8, out_len: *mut usize) -> i32 {
    guard(|| {
        if out_ptr.is_null() || out_len.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let json = recall_stats_json().into_bytes();
        // SAFETY: caller contract.
        unsafe { crate::ffm_bridge::give(json, out_ptr, out_len) };
        Ok(())
    })
}

/// The reader's top `k` for the request in `request` (see [`lucene_engine::knn`]):
/// global doc ids into `out_docs`, scores into `out_scores`, best first,
/// `*out_count` of them.
///
/// # Safety
/// `request` valid for `request_len` bytes; `out_docs`/`out_scores` for
/// `buf_len` elements each, with `buf_len >= k`; `out_count` for one write.
/// An expanded nested search may need more than `k` (see the
/// [`lucene_engine::knn`]).
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_knn(
    handle: u64,
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
        let request = decode(blob)?;
        if buf_len < request.k {
            return Err(FfiStatus::BufferTooSmall);
        }
        let h = lookup(
            handle,
            "ffi_jvm_reader_knn: unknown or already-closed handle",
        )?;
        let hits = search(&h, &request, &[])?;
        // SAFETY: caller contract.
        unsafe { write_hits(&hits, out_docs, out_scores, buf_len, out_count) }
    })
}

/// Writes `hits` out as [`ffi_jvm_reader_knn`] does, or the count they
/// need when they do not fit `buf_len`.
///
/// # Safety
/// As [`ffi_jvm_reader_knn`]'s outputs.
pub(crate) unsafe fn write_hits(
    hits: &[lucene_search::ScoreDoc],
    out_docs: *mut i32,
    out_scores: *mut f32,
    buf_len: usize,
    out_count: *mut usize,
) -> Result<(), FfiStatus> {
    if hits.len() > buf_len {
        // SAFETY: caller contract.
        unsafe { *out_count = hits.len() };
        set_last_error(format!(
            "KNN request: {} hits do not fit {buf_len}",
            hits.len()
        ));
        return Err(FfiStatus::BufferTooSmall);
    }
    // SAFETY: caller contract; `hits.len() <= buf_len`.
    unsafe {
        for (i, hit) in hits.iter().enumerate() {
            *out_docs.add(i) = hit.doc_id;
            *out_scores.add(i) = hit.score;
        }
        *out_count = hits.len();
    }
    Ok(())
}
