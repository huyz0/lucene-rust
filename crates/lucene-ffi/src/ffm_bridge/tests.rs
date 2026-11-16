#![cfg(test)]
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
    let rc =
        unsafe { ffi_jvm_reader_doc_freq(h, b"body".as_ptr(), 4, b"fox".as_ptr(), 3, &mut out) };
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
    let rc =
        unsafe { ffi_jvm_reader_doc_freq(h, std::ptr::null(), 4, b"fox".as_ptr(), 3, &mut out) };
    assert_eq!(rc, FfiStatus::NullPointer.code());
    let rc =
        unsafe { ffi_jvm_reader_doc_freq(h, b"body".as_ptr(), 4, std::ptr::null(), 3, &mut out) };
    assert_eq!(rc, FfiStatus::NullPointer.code());
    assert_eq!(ffi_close_jvm_reader(h), 0);
    let rc =
        unsafe { ffi_jvm_reader_doc_freq(h, b"body".as_ptr(), 4, b"fox".as_ptr(), 3, &mut out) };
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
    let (_, _, got_total, _) = aggregate(h, &q, &aggs, 0, 1, jvm_reader::METRIC_VALUES).unwrap();
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

#[test]
fn parent_cache_stats_report_and_refuse_a_null_out() {
    let mut out = [-1i64; 2];
    // SAFETY: two slots.
    assert_eq!(unsafe { ffi_jvm_parent_cache_stats(out.as_mut_ptr()) }, 0);
    assert!(out[0] >= 0 && out[1] >= 0);
    // SAFETY: null is checked.
    assert_eq!(
        unsafe { ffi_jvm_parent_cache_stats(std::ptr::null_mut()) },
        FfiStatus::NullPointer.code()
    );
}

#[test]
fn a_collapsed_search_hands_back_its_encoding() {
    let h = open();
    let q = term_blob("body", "fox");
    let sort = sort_blob_terms(&[(SORT_LONG, 0, "n", 0)], None);
    let mut collapse = 1i32.to_le_bytes().to_vec();
    collapse.extend_from_slice(b"n");
    collapse.push(0);
    let call = |collapse: &[u8], out_ptr: *mut *mut u8, out_len: *mut usize| unsafe {
        ffi_jvm_reader_search_collapsed_alloc(
            h,
            q.as_ptr(),
            q.len(),
            sort.as_ptr(),
            sort.len(),
            collapse.as_ptr(),
            collapse.len(),
            4,
            out_ptr,
            out_len,
        )
    };
    let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
    assert_eq!(call(&collapse, &mut p, &mut n), 0, "{}", last_error_text());
    assert_eq!(
        take(p, n),
        jvm_reader::search_collapsed_blobs(h, &q, &sort, &collapse, 4).unwrap()
    );
    let (mut p, mut n) = (std::ptr::null_mut(), 0usize);
    // A truncated collapse blob: refused, nothing handed back.
    assert_ne!(call(&collapse[..2], &mut p, &mut n), 0);
    assert!(p.is_null() && n == 0);
    assert_eq!(
        call(&collapse, std::ptr::null_mut(), &mut n),
        FfiStatus::NullPointer.code()
    );
    assert_eq!(ffi_close_jvm_reader(h), 0);
}
