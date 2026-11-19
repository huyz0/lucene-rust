//! The C-ABI functions the OpenSearch plugin's `NativeBridge` reaches through
//! Java's Foreign Function & Memory API (`java.lang.foreign`, JDK 22+; the
//! plugin targets JDK 25, which OpenSearch 3.9.0 bundles).
//!
//! **Most of the plugin's calls need nothing here.** A downcall reaches any
//! `extern "C"` function, so `NativeBridge` calls
//! [`jvm_reader::ffi_jvm_reader_search`], [`jvm_reader::ffi_open_jvm_reader`],
//! the `ffi_engine_writer_*` family and the rest directly, with arguments and
//! results in native memory it owns. What is here is what those functions do
//! not already offer:
//!
//! - results whose size is known only after the work is done (a sorted
//!   search's keyword terms, the aggregation results, a stored document),
//!   handed back as a buffer this crate allocated, which the caller copies
//!   and returns with [`ffi_jvm_free_bytes`] -- where the caller-sized C
//!   functions would make it guess a capacity and run the search again when
//!   the guess was short;
//! - [`ffi_jvm_reader_doc_freq`], which had only a JNI entry point;
//! - [`ffi_jvm_last_error`], the thread's last error as a buffer of its exact
//!   length.
//!
//! The contract is the `ffi-safety` one: an `i32` status (`FfiStatus`), every
//! body inside [`guard`], and a failure's message in the thread's last-error
//! slot. An owned buffer is written only on success; on failure `*out_ptr` is
//! null and `*out_len` 0, so there is nothing to free.

use crate::error::{guard, last_error, set_last_error, FfiStatus};
use crate::jvm_reader;
use crate::raw::bytes_from_raw;

/// Hands `bytes` to the caller: `*out_ptr`/`*out_len` describe a buffer it
/// must return through [`ffi_jvm_free_bytes`]. An empty result is a null
/// pointer, which needs no free.
///
/// # Safety
/// `out_ptr` and `out_len` must be valid for one write each.
pub(crate) unsafe fn give(bytes: Vec<u8>, out_ptr: *mut *mut u8, out_len: *mut usize) {
    let (ptr, len) = if bytes.is_empty() {
        (std::ptr::null_mut(), 0)
    } else {
        let len = bytes.len();
        (Box::into_raw(bytes.into_boxed_slice()).cast::<u8>(), len)
    };
    // SAFETY: caller contract.
    unsafe {
        *out_ptr = ptr;
        *out_len = len;
    }
}

/// Clears an owned-buffer out-parameter pair, so a failed call leaves nothing
/// to free; refuses null out-pointers.
///
/// # Safety
/// Non-null `out_ptr`/`out_len` must be valid for one write each.
pub(crate) unsafe fn clear_out(
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> Result<(), FfiStatus> {
    if out_ptr.is_null() || out_len.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    // SAFETY: caller contract; both checked non-null above.
    unsafe {
        *out_ptr = std::ptr::null_mut();
        *out_len = 0;
    }
    Ok(())
}

/// Returns a buffer this module handed out. A null `ptr` is a no-op.
///
/// # Safety
/// `ptr`/`len` must be exactly a pair this module wrote, not freed before.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_free_bytes(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: caller contract: `ptr` came from `Box::<[u8]>::into_raw` of
    // exactly `len` bytes in `give`. Dropping cannot unwind.
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
}

/// The calling thread's last error message, UTF-8, as an owned buffer (null
/// when there is none). The slot itself is left as it is.
///
/// # Safety
/// `out_ptr`/`out_len` must be valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_last_error(out_ptr: *mut *mut u8, out_len: *mut usize) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        return FfiStatus::NullPointer.code();
    }
    // Not under `guard`, which would record a message of its own on failure
    // and so overwrite the one being read. Reading a thread-local and copying
    // it can fail only by allocation failure, which aborts rather than
    // unwinds; the `catch_unwind` is the boundary rule, not an expectation.
    let message = std::panic::catch_unwind(|| last_error().into_bytes()).unwrap_or_default();
    // SAFETY: caller contract.
    unsafe { give(message, out_ptr, out_len) };
    FfiStatus::Ok.code()
}

/// Records `message` (UTF-8, lossily) as the calling thread's last error:
/// the plugin's own argument checks report through the same slot as the
/// native code's, so one `lastError()` reads either.
///
/// # Safety
/// `message` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_set_last_error(message: *const u8, len: usize) -> i32 {
    // SAFETY: caller contract.
    let bytes = match unsafe { bytes_from_raw(message, len) } {
        Ok(b) => b,
        Err(status) => return status.code(),
    };
    // Not under `guard`, which would clear the flag this sets. Formatting a
    // message can fail only by allocation failure, which aborts rather than
    // unwinds; the `catch_unwind` is the boundary rule.
    match std::panic::catch_unwind(|| set_last_error(String::from_utf8_lossy(bytes))) {
        Ok(()) => FfiStatus::Ok.code(),
        Err(_) => FfiStatus::Panic.code(),
    }
}

/// `IndexReader.docFreq(new Term(field, term))` over the reader's segments,
/// into `*out`.
///
/// # Safety
/// `field`/`term` must be valid for `field_len`/`term_len` bytes and `out`
/// for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_doc_freq(
    handle: u64,
    field: *const u8,
    field_len: usize,
    term: *const u8,
    term_len: usize,
    out: *mut i64,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let field = unsafe { bytes_from_raw(field, field_len)? };
        // SAFETY: caller contract.
        let term = unsafe { bytes_from_raw(term, term_len)? };
        let n = jvm_reader::doc_freq(handle, field, term)?;
        // SAFETY: caller contract.
        unsafe { *out = n };
        Ok(())
    })
}

/// The nested queries' parent bit-set cache
/// ([`lucene_engine::nodes::parent_cache_stats`]): its entries into `out[0]` and
/// the bytes of its sets into `out[1]`, for the plugin's stats.
///
/// # Safety
/// `out` must be valid for two `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_parent_cache_stats(out: *mut i64) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let (entries, bytes) = lucene_engine::nodes::parent_cache_stats();
        let to = |v: usize| i64::try_from(v).unwrap_or(i64::MAX);
        // SAFETY: caller contract (two slots).
        unsafe {
            *out = to(entries);
            *out.add(1) = to(bytes);
        }
        Ok(())
    })
}

/// Slots of [`ffi_jvm_reader_search_sorted_alloc`]'s `out_counts`.
pub const SORTED_COUNTS: usize = 6;

/// [`jvm_reader::ffi_jvm_reader_search_sorted`] with its keyword terms handed
/// back as an owned buffer: `out_docs` receives the hits' doc ids and
/// `out_values` their sort values (key-major per hit), `out_counts` the
/// [`SORTED_COUNTS`] slots `[hits, total, total is a lower bound, max score
/// bits, terminated, any key is a keyword key]`, and
/// `*out_terms`/`*out_terms_len` the keyword keys' terms
/// ([`jvm_reader::encode_terms`]; null when there are none -- the last count
/// slot says whether none is an empty answer or no answer). `top_n` must be
/// at least 1.
///
/// # Safety
/// `query`/`sort` must be valid for `query_len`/`sort_len` bytes, `out_docs`
/// for `docs_cap` and `out_values` for `values_cap` writes, `out_counts` for
/// [`SORTED_COUNTS`], and `out_terms`/`out_terms_len` for one each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_search_sorted_alloc(
    handle: u64,
    query: *const u8,
    query_len: usize,
    sort: *const u8,
    sort_len: usize,
    top_n: usize,
    count_limit: i64,
    out_docs: *mut i32,
    docs_cap: usize,
    out_values: *mut i64,
    values_cap: usize,
    out_counts: *mut i64,
    out_terms: *mut *mut u8,
    out_terms_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_terms, out_terms_len)? };
        if out_docs.is_null() || out_values.is_null() || out_counts.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let sort_blob = unsafe { bytes_from_raw(sort, sort_len)? };
        let keys = usize::from(sort_blob.first().copied().unwrap_or(0));
        let want_values = top_n.checked_mul(keys).ok_or(FfiStatus::InvalidArgument)?;
        if docs_cap < top_n || values_cap < want_values {
            set_last_error(format!(
                "output buffers hold {docs_cap} hits and {values_cap} values, topN is {top_n} with {keys} keys"
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        if top_n == 0 {
            set_last_error("searchSorted: topN must be at least 1");
            return Err(FfiStatus::InvalidArgument);
        }
        let out = jvm_reader::search_sorted_blobs(handle, blob, sort_blob, top_n, count_limit)?;
        // SAFETY: caller contract: `out_docs` holds `docs_cap >= top_n` and
        // `out_values` `values_cap >= top_n * keys` slots; the search returns
        // at most `top_n` hits of `out.keys` (the blob's key count) values.
        let (docs, values) = unsafe {
            (
                std::slice::from_raw_parts_mut(out_docs, docs_cap),
                std::slice::from_raw_parts_mut(out_values, values_cap),
            )
        };
        for (i, hit) in out.hits.iter().enumerate() {
            docs[i] = hit.doc;
            for (k, &v) in hit.values.iter().take(out.keys).enumerate() {
                values[i.saturating_mul(out.keys).saturating_add(k)] = v;
            }
        }
        let counts = [
            out.hits.len() as i64,
            out.total,
            i64::from(out.lower_bound),
            i64::from(out.max_score.to_bits()),
            i64::from(out.terminated),
            i64::from(out.has_terms),
        ];
        // SAFETY: caller contract: `out_counts` holds `SORTED_COUNTS` slots.
        unsafe { std::ptr::copy_nonoverlapping(counts.as_ptr(), out_counts, SORTED_COUNTS) };
        // SAFETY: caller contract.
        unsafe { give(out.terms, out_terms, out_terms_len) };
        Ok(())
    })
}

/// Slots of [`ffi_jvm_reader_aggregate_alloc`]'s `out_total`.
pub const AGGREGATE_TOTAL: usize = 4;

/// [`jvm_reader::ffi_jvm_reader_aggregate`] with its `terms` results handed
/// back as an owned buffer: per metric field its value count in
/// `out_counts` and [`jvm_reader::METRIC_VALUES`] doubles in `out_values`;
/// with a positive `count_limit`, `out_total` the [`AGGREGATE_TOTAL`] slots
/// `[1 when counted else 0, total, total is a lower bound, 1 when
/// out_seg_counts was written else 0]`. When counted and every segment's
/// live matches were visited, and `out_seg_counts` holds a slot per segment,
/// those counts go there (by segment, in reader order): what the count
/// collector would iterate in each, so the caller can replay its early
/// termination (`terminated_early`) without a second search.
///
/// # Safety
/// `query`/`aggs` must be valid for `query_len`/`aggs_len` bytes,
/// `out_counts` for `counts_cap` and `out_values` for `values_cap` writes,
/// `out_total` for [`AGGREGATE_TOTAL`], `out_seg_counts` for `seg_cap` (null
/// when 0), and `out_terms`/`out_terms_len` for one each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_aggregate_alloc(
    handle: u64,
    query: *const u8,
    query_len: usize,
    aggs: *const u8,
    aggs_len: usize,
    count_limit: i64,
    out_counts: *mut i64,
    counts_cap: usize,
    out_values: *mut f64,
    values_cap: usize,
    out_total: *mut i64,
    out_seg_counts: *mut i64,
    seg_cap: usize,
    out_terms: *mut *mut u8,
    out_terms_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_terms, out_terms_len)? };
        if out_counts.is_null()
            || out_values.is_null()
            || out_total.is_null()
            || (out_seg_counts.is_null() && seg_cap > 0)
        {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let aggs_blob = unsafe { bytes_from_raw(aggs, aggs_len)? };
        let (states, terms, total, seen) =
            jvm_reader::aggregate_counting_blobs_seen(handle, blob, aggs_blob, count_limit)?;
        // Every segment's count, when the pass visited them all.
        let per_segment: Option<Vec<u64>> = seen.iter().copied().collect();
        let per_segment = per_segment.filter(|c| total.is_some() && c.len() <= seg_cap);
        if let Some(c) = &per_segment {
            for (i, &n) in c.iter().enumerate() {
                // SAFETY: caller contract: `out_seg_counts` holds `seg_cap >=
                // c.len()` slots.
                unsafe { *out_seg_counts.add(i) = i64::try_from(n).unwrap_or(i64::MAX) };
            }
        }
        let written = i64::from(per_segment.is_some());
        let totals = match total {
            Some((total, lower_bound)) => [1, total, i64::from(lower_bound), written],
            None => [0, 0, 0, 0],
        };
        // SAFETY: caller contract: `out_total` holds `AGGREGATE_TOTAL` slots.
        unsafe { std::ptr::copy_nonoverlapping(totals.as_ptr(), out_total, AGGREGATE_TOTAL) };
        let want_values = states.len().saturating_mul(jvm_reader::METRIC_VALUES);
        if counts_cap < states.len() || values_cap < want_values {
            set_last_error(format!(
                "output buffers hold {counts_cap} counts and {values_cap} values for {} fields",
                states.len()
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        // SAFETY: caller contract, and the capacities checked just above.
        let (counts, values) = unsafe {
            (
                std::slice::from_raw_parts_mut(out_counts, counts_cap),
                std::slice::from_raw_parts_mut(out_values, values_cap),
            )
        };
        for (i, s) in states.iter().enumerate() {
            counts[i] = i64::try_from(s.count).unwrap_or(i64::MAX);
            let v = jvm_reader::metric_values(s);
            let from = i.saturating_mul(jvm_reader::METRIC_VALUES);
            values[from..from.saturating_add(jvm_reader::METRIC_VALUES)].copy_from_slice(&v);
        }
        // SAFETY: caller contract.
        unsafe { give(terms, out_terms, out_terms_len) };
        Ok(())
    })
}

/// [`crate::jvm_aggs::ffi_jvm_reader_aggregate_tree`], the encoded results as
/// an owned buffer.
///
/// # Safety
/// `query`/`tree` must be valid for `query_len`/`tree_len` bytes and
/// `out_ptr`/`out_len` for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_aggregate_tree_alloc(
    handle: u64,
    query: *const u8,
    query_len: usize,
    tree: *const u8,
    tree_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let tree_blob = unsafe { bytes_from_raw(tree, tree_len)? };
        let encoded = crate::jvm_aggs::aggregate_tree_blobs(handle, blob, tree_blob)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`ffi_jvm_reader_aggregate_tree_alloc`], each committed segment's
/// collected part read from and kept in the process's aggregation segment
/// cache ([`crate::jvm_aggs::ffi_agg_segment_cache_set_budget`]): the same
/// results, and with the cache off the same pass.
///
/// # Safety
/// As [`ffi_jvm_reader_aggregate_tree_alloc`].
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_aggregate_tree_cached_alloc(
    handle: u64,
    query: *const u8,
    query_len: usize,
    tree: *const u8,
    tree_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let tree_blob = unsafe { bytes_from_raw(tree, tree_len)? };
        let encoded = crate::jvm_aggs::aggregate_tree_blobs_cached(handle, blob, tree_blob, true)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`jvm_reader::search_hybrid_blobs`]: a hybrid query's shard phase, its
/// slices' answers encoded in an owned buffer.
///
/// # Safety
/// `blob` must be valid for `blob_len` bytes and `out_ptr`/`out_len` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_search_hybrid_alloc(
    handle: u64,
    blob: *const u8,
    blob_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(blob, blob_len)? };
        let encoded = jvm_reader::search_hybrid_blobs(handle, blob)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`jvm_reader::profile_blobs`]: a request's query run once more with a timer around each
/// clause (`profile`), the timers encoded in an owned buffer.
///
/// # Safety
/// `blob` must be valid for `blob_len` bytes and `out_ptr`/`out_len` for one
/// write each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_profile_alloc(
    handle: u64,
    blob: *const u8,
    blob_len: usize,
    collect: i32,
    top_n: usize,
    count_limit: i64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(blob, blob_len)? };
        let encoded = jvm_reader::profile_blobs(
            handle,
            blob,
            u8::try_from(collect).unwrap_or(u8::MAX),
            top_n,
            count_limit,
        )?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`jvm_reader::search_collapsed_blobs`]: a collapsed search (OpenSearch's
/// `collapse`), its slices' groups encoded in an owned buffer.
///
/// # Safety
/// `query`/`sort`/`collapse` must be valid for `query_len`/`sort_len`/
/// `collapse_len` bytes and `out_ptr`/`out_len` for one write each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_reader_search_collapsed_alloc(
    handle: u64,
    query: *const u8,
    query_len: usize,
    sort: *const u8,
    sort_len: usize,
    collapse: *const u8,
    collapse_len: usize,
    top_n: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(query, query_len)? };
        // SAFETY: caller contract.
        let sort_blob = unsafe { bytes_from_raw(sort, sort_len)? };
        // SAFETY: caller contract.
        let collapse_blob = unsafe { bytes_from_raw(collapse, collapse_len)? };
        let encoded =
            jvm_reader::search_collapsed_blobs(handle, blob, sort_blob, collapse_blob, top_n)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`crate::jvm_fetch::ffi_jvm_reader_document`], the encoded fields as an
/// owned buffer.
///
/// # Safety
/// `out_ptr`/`out_len` must be valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_document_alloc(
    handle: u64,
    segment: i32,
    doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        let encoded = crate::jvm_fetch::document_blob(handle, segment, doc)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`crate::jvm_fetch::doc_values_blob`]: segment `segment`'s document
/// `doc` -- its values of the `count` fields numbered at `fields` -- as an
/// owned buffer.
///
/// # Safety
/// `fields` must be valid for `count` `i32` reads (null when 0);
/// `out_ptr`/`out_len` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_doc_values_alloc(
    handle: u64,
    segment: i32,
    doc: i32,
    fields: *const i32,
    count: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        if fields.is_null() && count > 0 {
            return Err(FfiStatus::NullPointer);
        }
        let fields: &[i32] = if count == 0 {
            &[]
        } else {
            // SAFETY: caller contract, non-null checked above.
            unsafe { std::slice::from_raw_parts(fields, count) }
        };
        let encoded = crate::jvm_fetch::doc_values_blob(handle, segment, doc, fields)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`crate::jvm_fetch::source_blob`]: a derived-source document's `_source`, its vectors put
/// back, as an owned buffer.
///
/// # Safety
/// `out_ptr`/`out_len` must be valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_derived_source_alloc(
    handle: u64,
    segment: i32,
    doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        let encoded = crate::jvm_fetch::source_blob(handle, segment, doc, true)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`crate::jvm_fetch::source_blob`] without the vectors: a derived-source document's `_source`
/// as stored, each vector masked, as an owned buffer.
///
/// # Safety
/// `out_ptr`/`out_len` must be valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_stored_source_alloc(
    handle: u64,
    segment: i32,
    doc: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        let encoded = crate::jvm_fetch::source_blob(handle, segment, doc, false)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`crate::jvm_fetch::stored_value`]: the value a document blob left out as `DEFERRED`, as an
/// owned buffer.
///
/// # Safety
/// `out_ptr`/`out_len` must be valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_reader_stored_value_alloc(
    handle: u64,
    segment: i32,
    doc: i32,
    ordinal: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        let encoded = crate::jvm_fetch::stored_value(handle, segment, doc, ordinal)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

/// [`lucene_engine::analysis::register`]: an analyzer for indexing, its id into
/// `*out_id`.
///
/// # Safety
/// `blob` must be valid for `blob_len` bytes and `out_id` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_analysis_register(
    blob: *const u8,
    blob_len: usize,
    out_id: *mut i32,
) -> i32 {
    guard(|| {
        if out_id.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(blob, blob_len)? };
        let id = lucene_engine::analysis::register(blob)?;
        // SAFETY: caller contract.
        unsafe { *out_id = i32::try_from(id).map_err(|_| FfiStatus::InvalidArgument)? };
        Ok(())
    })
}

/// [`lucene_engine::ingest::register`]: a mapping's field plan, its id into `*out_id`.
///
/// # Safety
/// `blob` must be valid for `blob_len` bytes and `out_id` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_ingest_register(
    blob: *const u8,
    blob_len: usize,
    out_id: *mut i32,
) -> i32 {
    guard(|| {
        if out_id.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(blob, blob_len)? };
        let id = lucene_engine::ingest::register(blob)?;
        // SAFETY: caller contract.
        unsafe { *out_id = i32::try_from(id).map_err(|_| FfiStatus::InvalidArgument)? };
        Ok(())
    })
}

/// [`lucene_engine::ingest::parse`]: `src` read against plan `plan`. `*out_len` is 0
/// when the document is refused (the JVM's mapper takes it), else the fragment's
/// length, which is in `out` when it is at most `out_cap` and otherwise not written.
///
/// # Safety
/// `src` valid for `src_len` bytes, `out` for `out_cap`, `out_len` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_ingest_parse(
    plan: i32,
    src: *const u8,
    src_len: usize,
    out: *mut u8,
    out_cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() || (out.is_null() && out_cap != 0) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        unsafe { *out_len = 0 };
        let plan = u32::try_from(plan)
            .ok()
            .and_then(lucene_engine::ingest::plan)
            .ok_or(FfiStatus::InvalidHandle)?;
        // SAFETY: caller contract.
        let src = unsafe { bytes_from_raw(src, src_len)? };
        thread_local! {
            static FRAGMENT: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        FRAGMENT.with(|f| {
            let mut f = f.borrow_mut();
            if lucene_engine::ingest::parse(&plan, src, &mut f) {
                // SAFETY: caller contract.
                unsafe {
                    *out_len = f.len();
                    if f.len() <= out_cap {
                        std::ptr::copy_nonoverlapping(f.as_ptr(), out, f.len());
                    }
                }
            }
        });
        Ok(())
    })
}

/// [`lucene_engine::ingest::verify`]: two single-document operation blobs compared as
/// decoded documents; `*out_len` is 0 when they agree, else the length of the difference
/// written (truncated to `out_cap`) into `out`.
///
/// # Safety
/// Each pointer valid for its length, `out_len` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_ingest_verify(
    expected: *const u8,
    expected_len: usize,
    actual: *const u8,
    actual_len: usize,
    out: *mut u8,
    out_cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() || (out.is_null() && out_cap != 0) {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let (expected, actual) = unsafe {
            (
                bytes_from_raw(expected, expected_len)?,
                bytes_from_raw(actual, actual_len)?,
            )
        };
        let text = lucene_engine::ingest::verify(expected, actual)?.unwrap_or_default();
        let n = text.len().min(out_cap);
        // SAFETY: caller contract.
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), out, n);
            *out_len = n;
        }
        Ok(())
    })
}

/// [`lucene_engine::painless_score::check`]: whether the native runtime runs a
/// score script exactly as Painless would, decided before anything runs.
/// Writes nothing (a null buffer) when it does, else the reason as UTF-8.
///
/// # Safety
/// `blob` must be valid for `blob_len` bytes, and `out_ptr`/`out_len` for
/// one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_painless_check(
    blob: *const u8,
    blob_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let blob = unsafe { bytes_from_raw(blob, blob_len)? };
        let refused = match lucene_engine::painless_score::check(blob)? {
            None => Vec::new(),
            Some(r) if r.is_empty() => b"refused".to_vec(),
            Some(r) => r.into_bytes(),
        };
        // SAFETY: caller contract.
        unsafe { give(refused, out_ptr, out_len) };
        Ok(())
    })
}

/// [`lucene_engine::analysis::analyze`]: `spec`'s analyzer over `text` (UTF-8)
/// as field `field`, its tokens into an owned buffer.
///
/// # Safety
/// `spec`/`field`/`text` must be valid for their lengths, and
/// `out_ptr`/`out_len` for one write each.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_jvm_analyze_alloc(
    spec: *const u8,
    spec_len: usize,
    field: *const u8,
    field_len: usize,
    text: *const u8,
    text_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let spec = unsafe { bytes_from_raw(spec, spec_len)? };
        // SAFETY: caller contract.
        let field = unsafe { bytes_from_raw(field, field_len)? };
        // SAFETY: caller contract.
        let text = unsafe { bytes_from_raw(text, text_len)? };
        let utf8 = |b| std::str::from_utf8(b).map_err(|_| FfiStatus::InvalidUtf8);
        let (field, text) = (utf8(field)?, utf8(text)?);
        let encoded = lucene_engine::analysis::analyze(spec, field, text)?;
        // SAFETY: caller contract.
        unsafe { give(encoded, out_ptr, out_len) };
        Ok(())
    })
}

#[cfg(test)]
mod tests;
