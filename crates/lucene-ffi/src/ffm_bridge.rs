//! The C-ABI functions the OpenSearch plugin's `NativeBridge` reaches through
//! Java's Foreign Function & Memory API (`java.lang.foreign`, JDK 22+; the
//! plugin targets JDK 25, which OpenSearch 3.8.0 bundles).
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
unsafe fn give(bytes: Vec<u8>, out_ptr: *mut *mut u8, out_len: *mut usize) {
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
unsafe fn clear_out(out_ptr: *mut *mut u8, out_len: *mut usize) -> Result<(), FfiStatus> {
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
                values[i * out.keys + k] = v;
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
            values[i * jvm_reader::METRIC_VALUES..(i + 1) * jvm_reader::METRIC_VALUES]
                .copy_from_slice(&v);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_reader::tests::{aggs_blob, open, sort_blob_terms, term_blob};
    use crate::jvm_reader::{ffi_close_jvm_reader, SORT_LONG, SORT_STRING};

    /// Copies an owned buffer out and frees it.
    fn take(ptr: *mut u8, len: usize) -> Vec<u8> {
        if ptr.is_null() {
            assert_eq!(len, 0);
            return Vec::new();
        }
        let v = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        unsafe { ffi_jvm_free_bytes(ptr, len) };
        v
    }

    fn last_error_text() -> String {
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        assert_eq!(unsafe { ffi_jvm_last_error(&mut p, &mut n) }, 0);
        String::from_utf8(take(p, n)).unwrap()
    }

    #[test]
    fn last_error_reads_the_slot_without_changing_it() {
        set_last_error("the message");
        assert_eq!(last_error_text(), "the message");
        assert_eq!(last_error_text(), "the message");
        set_last_error("");
        assert_eq!(last_error_text(), "");
        let m = "from Java \u{e9}";
        assert_eq!(unsafe { ffi_jvm_set_last_error(m.as_ptr(), m.len()) }, 0);
        assert_eq!(last_error_text(), m);
        assert_eq!(unsafe { ffi_jvm_set_last_error(b"\xff".as_ptr(), 1) }, 0);
        assert_eq!(last_error_text(), "\u{fffd}");
        assert_eq!(
            unsafe { ffi_jvm_set_last_error(std::ptr::null(), 1) },
            FfiStatus::NullPointer.code()
        );
        let mut n = 0usize;
        assert_eq!(
            unsafe { ffi_jvm_last_error(std::ptr::null_mut(), &mut n) },
            FfiStatus::NullPointer.code()
        );
        // Freeing null is a no-op.
        unsafe { ffi_jvm_free_bytes(std::ptr::null_mut(), 0) };
    }

    #[test]
    fn a_message_set_from_java_is_never_a_later_failures_diagnosis() {
        let m = "refused in Java";
        assert_eq!(unsafe { ffi_jvm_set_last_error(m.as_ptr(), m.len()) }, 0);
        // A guarded call failing without recording a message of its own gets
        // its status's default message, not the one Java left in the slot.
        let rc = unsafe {
            ffi_jvm_reader_doc_freq(0, b"f".as_ptr(), 1, b"t".as_ptr(), 1, std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        assert_eq!(last_error_text(), FfiStatus::NullPointer.default_message());
    }

    #[test]
    fn doc_freq_counts_a_term_and_reports_failures() {
        let h = open();
        let mut out = -1i64;
        let rc = unsafe {
            ffi_jvm_reader_doc_freq(h, b"body".as_ptr(), 4, b"fox".as_ptr(), 3, &mut out)
        };
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(Ok(out), jvm_reader::doc_freq(h, b"body", b"fox"));
        assert!(out > 0);
        let rc = unsafe {
            ffi_jvm_reader_doc_freq(
                h,
                b"body".as_ptr(),
                4,
                b"fox".as_ptr(),
                3,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_jvm_reader_doc_freq(h, std::ptr::null(), 4, b"fox".as_ptr(), 3, &mut out)
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_jvm_reader_doc_freq(h, b"body".as_ptr(), 4, std::ptr::null(), 3, &mut out)
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        assert_eq!(ffi_close_jvm_reader(h), 0);
        let rc = unsafe {
            ffi_jvm_reader_doc_freq(h, b"body".as_ptr(), 4, b"fox".as_ptr(), 3, &mut out)
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        assert!(!last_error_text().is_empty());
    }

    /// `(docs, values, counts, terms)` of a sorted search, or the status.
    type Sorted = (Vec<i32>, Vec<i64>, [i64; SORTED_COUNTS], Vec<u8>);

    fn sorted(
        h: u64,
        query: &[u8],
        sort: &[u8],
        top_n: usize,
        docs_cap: usize,
        values_cap: usize,
    ) -> Result<Sorted, i32> {
        let mut docs = vec![0i32; docs_cap];
        let mut values = vec![0i64; values_cap];
        let mut counts = [0i64; SORTED_COUNTS];
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        let rc = unsafe {
            ffi_jvm_reader_search_sorted_alloc(
                h,
                query.as_ptr(),
                query.len(),
                sort.as_ptr(),
                sort.len(),
                top_n,
                i64::MAX,
                docs.as_mut_ptr(),
                docs_cap,
                values.as_mut_ptr(),
                values_cap,
                counts.as_mut_ptr(),
                &mut p,
                &mut n,
            )
        };
        if rc != 0 {
            assert!(p.is_null() && n == 0, "a failure hands back nothing");
            return Err(rc);
        }
        Ok((docs, values, counts, take(p, n)))
    }

    #[test]
    fn a_sorted_search_matches_the_caller_sized_entry_point() {
        let h = open();
        let q = term_blob("body", "fox");
        // A long key: no terms come back.
        let long_sort = sort_blob_terms(&[(SORT_LONG, 0, "n", 0)], None);
        let (docs, values, counts, terms) = sorted(h, &q, &long_sort, 4, 4, 4).unwrap();
        let expected = jvm_reader::search_sorted_blobs(h, &q, &long_sort, 4, i64::MAX).unwrap();
        let hits = expected.hits.len();
        assert!(hits > 0);
        assert_eq!(counts[0] as usize, hits);
        assert_eq!(counts[1], expected.total);
        assert_eq!(counts[2], i64::from(expected.lower_bound));
        assert_eq!(counts[3], i64::from(expected.max_score.to_bits()));
        assert_eq!(counts[4], i64::from(expected.terminated));
        assert_eq!(counts[5], 0, "no keyword key");
        for (i, hit) in expected.hits.iter().enumerate() {
            assert_eq!(docs[i], hit.doc);
            assert_eq!(values[i], hit.values[0]);
        }
        assert!(terms.is_empty());
        // A keyword key: the terms come back, exactly as encoded.
        let kw_sort = sort_blob_terms(&[(SORT_STRING, 0, "body", 0)], None);
        let expected = jvm_reader::search_sorted_blobs(h, &q, &kw_sort, 4, i64::MAX).unwrap();
        let (_, _, counts, terms) = sorted(h, &q, &kw_sort, 4, 4, 4).unwrap();
        assert_eq!(counts[5], 1, "a keyword key");
        assert!(!expected.terms.is_empty());
        assert_eq!(terms, expected.terms);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn a_sorted_search_refuses_bad_buffers() {
        let h = open();
        let q = term_blob("body", "fox");
        let s = sort_blob_terms(&[(SORT_LONG, 0, "n", 0)], None);
        assert_eq!(
            sorted(h, &q, &s, 4, 3, 4).unwrap_err(),
            FfiStatus::BufferTooSmall.code()
        );
        assert_eq!(
            sorted(h, &q, &s, 4, 4, 3).unwrap_err(),
            FfiStatus::BufferTooSmall.code()
        );
        assert_eq!(
            sorted(h, &q, &s, 0, 4, 4).unwrap_err(),
            FfiStatus::InvalidArgument.code()
        );
        assert_eq!(
            sorted(h, &q, &s, usize::MAX, 4, 4).unwrap_err(),
            FfiStatus::BufferTooSmall.code()
        );
        let mut counts = [0i64; SORTED_COUNTS];
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        let null_docs = unsafe {
            ffi_jvm_reader_search_sorted_alloc(
                h,
                q.as_ptr(),
                q.len(),
                s.as_ptr(),
                s.len(),
                1,
                0,
                std::ptr::null_mut(),
                1,
                [0i64; 1].as_mut_ptr(),
                1,
                counts.as_mut_ptr(),
                &mut p,
                &mut n,
            )
        };
        assert_eq!(null_docs, FfiStatus::NullPointer.code());
        let null_terms = unsafe {
            ffi_jvm_reader_search_sorted_alloc(
                h,
                q.as_ptr(),
                q.len(),
                s.as_ptr(),
                s.len(),
                1,
                0,
                [0i32; 1].as_mut_ptr(),
                1,
                [0i64; 1].as_mut_ptr(),
                1,
                counts.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut n,
            )
        };
        assert_eq!(null_terms, FfiStatus::NullPointer.code());
        let null_query = unsafe {
            ffi_jvm_reader_search_sorted_alloc(
                h,
                std::ptr::null(),
                3,
                s.as_ptr(),
                s.len(),
                1,
                0,
                [0i32; 1].as_mut_ptr(),
                1,
                [0i64; 1].as_mut_ptr(),
                1,
                counts.as_mut_ptr(),
                &mut p,
                &mut n,
            )
        };
        assert_eq!(null_query, FfiStatus::NullPointer.code());
        let null_sort = unsafe {
            ffi_jvm_reader_search_sorted_alloc(
                h,
                q.as_ptr(),
                q.len(),
                std::ptr::null(),
                3,
                1,
                0,
                [0i32; 1].as_mut_ptr(),
                1,
                [0i64; 1].as_mut_ptr(),
                1,
                counts.as_mut_ptr(),
                &mut p,
                &mut n,
            )
        };
        assert_eq!(null_sort, FfiStatus::NullPointer.code());
        assert_eq!(ffi_close_jvm_reader(h), 0);
        assert_eq!(
            sorted(h, &q, &s, 1, 1, 1).unwrap_err(),
            FfiStatus::InvalidHandle.code()
        );
    }

    /// `(counts, values, total, terms)` of an aggregation, or the status.
    type Aggregated = (Vec<i64>, Vec<f64>, [i64; AGGREGATE_TOTAL], Vec<u8>);

    fn aggregate(
        h: u64,
        query: &[u8],
        aggs: &[u8],
        limit: i64,
        counts_cap: usize,
        values_cap: usize,
    ) -> Result<Aggregated, i32> {
        let mut counts = vec![0i64; counts_cap];
        let mut values = vec![0f64; values_cap];
        let mut total = [-1i64; AGGREGATE_TOTAL];
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        let rc = unsafe {
            ffi_jvm_reader_aggregate_alloc(
                h,
                query.as_ptr(),
                query.len(),
                aggs.as_ptr(),
                aggs.len(),
                limit,
                counts.as_mut_ptr(),
                counts_cap,
                values.as_mut_ptr(),
                values_cap,
                total.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
                &mut p,
                &mut n,
            )
        };
        if rc != 0 {
            assert!(p.is_null() && n == 0, "a failure hands back nothing");
            return Err(rc);
        }
        Ok((counts, values, total, take(p, n)))
    }

    /// The segments' match counts come back when there is room for them and
    /// the total was counted; not otherwise.
    #[test]
    fn aggregate_hands_back_the_segments_counts() {
        let h = open();
        let q = term_blob("body", "fox");
        let aggs = aggs_blob(&[(0, 0, "n")], &[], &[]);
        let (_, _, total, seen) =
            jvm_reader::aggregate_counting_blobs_seen(h, &q, &aggs, i64::MAX).unwrap();
        let call = |limit: i64, seg: &mut [i64]| {
            let mut counts = [0i64; 1];
            let mut values = [0f64; jvm_reader::METRIC_VALUES];
            let mut out = [-1i64; AGGREGATE_TOTAL];
            let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
            let rc = unsafe {
                ffi_jvm_reader_aggregate_alloc(
                    h,
                    q.as_ptr(),
                    q.len(),
                    aggs.as_ptr(),
                    aggs.len(),
                    limit,
                    counts.as_mut_ptr(),
                    1,
                    values.as_mut_ptr(),
                    jvm_reader::METRIC_VALUES,
                    out.as_mut_ptr(),
                    seg.as_mut_ptr(),
                    seg.len(),
                    &mut p,
                    &mut n,
                )
            };
            assert_eq!(rc, 0);
            drop(take(p, n));
            out
        };
        let mut seg = vec![-1i64; seen.len()];
        let (t, lb) = total.unwrap();
        assert_eq!(call(i64::MAX, &mut seg), [1, t, i64::from(lb), 1]);
        let want: Vec<i64> = seen.iter().map(|c| c.unwrap() as i64).collect();
        assert_eq!(seg, want);
        assert_eq!(seg.iter().sum::<i64>(), t);
        // too little room, or no count asked for: nothing written
        let mut short = vec![-1i64; seen.len() - 1];
        assert_eq!(call(i64::MAX, &mut short)[3], 0);
        assert!(short.iter().all(|&c| c == -1));
        let mut seg = vec![-1i64; seen.len()];
        assert_eq!(call(0, &mut seg), [0, 0, 0, 0]);
        assert!(seg.iter().all(|&c| c == -1));
        // a null buffer with room claimed is refused
        let mut out = [0i64; AGGREGATE_TOTAL];
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        let rc = unsafe {
            ffi_jvm_reader_aggregate_alloc(
                h,
                q.as_ptr(),
                q.len(),
                aggs.as_ptr(),
                aggs.len(),
                0,
                [0i64; 1].as_mut_ptr(),
                1,
                [0f64; jvm_reader::METRIC_VALUES].as_mut_ptr(),
                jvm_reader::METRIC_VALUES,
                out.as_mut_ptr(),
                std::ptr::null_mut(),
                1,
                &mut p,
                &mut n,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn aggregate_matches_the_blob_level_results() {
        let h = open();
        let q = term_blob("body", "fox");
        let aggs = aggs_blob(&[(0, 0, "n")], &[("body", 10)], &[]);
        let (states, terms, total) =
            jvm_reader::aggregate_counting_blobs(h, &q, &aggs, i64::MAX).unwrap();
        let (counts, values, got_total, got_terms) =
            aggregate(h, &q, &aggs, i64::MAX, 1, jvm_reader::METRIC_VALUES).unwrap();
        assert_eq!(counts[0], states[0].count as i64);
        assert_eq!(values, jvm_reader::metric_values(&states[0]));
        assert_eq!(got_terms, terms);
        let (t, lb) = total.unwrap();
        // no room for the segments' counts: not written
        assert_eq!(got_total, [1, t, i64::from(lb), 0]);
        // No count asked for: the total slots say so.
        let (_, _, got_total, _) =
            aggregate(h, &q, &aggs, 0, 1, jvm_reader::METRIC_VALUES).unwrap();
        assert_eq!(got_total[0], 0);
        // Short buffers.
        assert_eq!(
            aggregate(h, &q, &aggs, 0, 0, 6).unwrap_err(),
            FfiStatus::BufferTooSmall.code()
        );
        assert_eq!(
            aggregate(h, &q, &aggs, 0, 1, 5).unwrap_err(),
            FfiStatus::BufferTooSmall.code()
        );
        // The count is written before the buffers are checked: the Java side
        // hands it back even when the rest is refused.
        let mut total = [0i64; AGGREGATE_TOTAL];
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        let rc = unsafe {
            ffi_jvm_reader_aggregate_alloc(
                h,
                q.as_ptr(),
                q.len(),
                aggs.as_ptr(),
                aggs.len(),
                i64::MAX,
                [0i64; 1].as_mut_ptr(),
                0,
                [0f64; 6].as_mut_ptr(),
                6,
                total.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
                &mut p,
                &mut n,
            )
        };
        assert_eq!(rc, FfiStatus::BufferTooSmall.code());
        assert!(p.is_null() && n == 0, "a failure hands back nothing");
        assert_eq!(total[0], 1, "counted before the refusal");
        // Null pointers.
        let mut total = [0i64; AGGREGATE_TOTAL];
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        let rc = unsafe {
            ffi_jvm_reader_aggregate_alloc(
                h,
                q.as_ptr(),
                q.len(),
                aggs.as_ptr(),
                aggs.len(),
                0,
                std::ptr::null_mut(),
                1,
                [0f64; 6].as_mut_ptr(),
                6,
                total.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
                &mut p,
                &mut n,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_jvm_reader_aggregate_alloc(
                h,
                q.as_ptr(),
                q.len(),
                aggs.as_ptr(),
                aggs.len(),
                0,
                [0i64; 1].as_mut_ptr(),
                1,
                [0f64; 6].as_mut_ptr(),
                6,
                total.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
                &mut p,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_jvm_reader_aggregate_alloc(
                h,
                std::ptr::null(),
                1,
                aggs.as_ptr(),
                aggs.len(),
                0,
                [0i64; 1].as_mut_ptr(),
                1,
                [0f64; 6].as_mut_ptr(),
                6,
                total.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
                &mut p,
                &mut n,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_jvm_reader_aggregate_alloc(
                h,
                q.as_ptr(),
                q.len(),
                std::ptr::null(),
                1,
                0,
                [0i64; 1].as_mut_ptr(),
                1,
                [0f64; 6].as_mut_ptr(),
                6,
                total.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
                &mut p,
                &mut n,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        assert_eq!(ffi_close_jvm_reader(h), 0);
        assert_eq!(
            aggregate(h, &q, &aggs, 0, 1, 6).unwrap_err(),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn aggregate_tree_and_document_hand_back_their_encodings() {
        let h = open();
        let q = term_blob("body", "fox");
        let tree = crate::jvm_aggs::tests::every_kind(false);
        let expected = crate::jvm_aggs::aggregate_tree_blobs(h, &q, &tree).unwrap();
        let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
        let rc = unsafe {
            ffi_jvm_reader_aggregate_tree_alloc(
                h,
                q.as_ptr(),
                q.len(),
                tree.as_ptr(),
                tree.len(),
                &mut p,
                &mut n,
            )
        };
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(take(p, n), expected);
        let rc = unsafe {
            ffi_jvm_reader_aggregate_tree_alloc(
                h,
                std::ptr::null(),
                1,
                tree.as_ptr(),
                tree.len(),
                &mut p,
                &mut n,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_jvm_reader_aggregate_tree_alloc(
                h,
                q.as_ptr(),
                q.len(),
                std::ptr::null(),
                1,
                &mut p,
                &mut n,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_jvm_reader_aggregate_tree_alloc(
                h,
                q.as_ptr(),
                q.len(),
                tree.as_ptr(),
                tree.len(),
                std::ptr::null_mut(),
                &mut n,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe {
            ffi_jvm_reader_aggregate_tree_alloc(
                h,
                q.as_ptr(),
                q.len(),
                b"\xff".as_ptr(),
                1,
                &mut p,
                &mut n,
            )
        };
        assert_ne!(rc, 0);
        assert!(p.is_null() && n == 0);

        let expected = crate::jvm_fetch::document_blob(h, 0, 0).unwrap();
        let rc = unsafe { ffi_jvm_reader_document_alloc(h, 0, 0, &mut p, &mut n) };
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(take(p, n), expected);
        let rc = unsafe { ffi_jvm_reader_document_alloc(h, 0, 0, &mut p, std::ptr::null_mut()) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        let rc = unsafe { ffi_jvm_reader_document_alloc(h, 9, 0, &mut p, &mut n) };
        assert_ne!(rc, 0);
        assert!(p.is_null() && n == 0);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }
}
