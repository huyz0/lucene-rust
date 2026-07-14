//! `ffi_assemble_fragments` (Highlighter FFI exposure): wraps this port's
//! existing `lucene_search::highlighter::assemble_fragments` across the FFI
//! boundary. No fragment-assembly logic is reimplemented here -- this
//! function is a thin marshal-in/call/marshal-out wrapper around
//! `crates/lucene-search/src/highlighter.rs`'s existing public API, following
//! the exact handle/error-code conventions `facets.rs`/`sort.rs` already
//! established.
//!
//! ## Inputs: caller-supplied text and spans, no segment/index state at all
//!
//! Unlike every other function in this crate, `ffi_assemble_fragments` needs
//! no `SegmentHandle` (or any other opened-index state): `assemble_fragments`
//! itself takes only a field's full text plus a set of already-computed
//! [`lucene_search::term_vectors_query::TermOffsetSpan`]s -- both of which
//! the caller must already have in hand (the text from wherever it fetched
//! stored field text, the spans from a prior
//! `lucene_search::term_vectors_query::matched_term_offsets` call, itself not
//! yet FFI-exposed -- see this module's doc comment on `docs/parity.md` for
//! that scope note). This function therefore takes `full_text` and the spans
//! as plain input buffers, not a handle lookup.
//!
//! **Span wire encoding**: `spans_count` spans are described by four parallel
//! arrays -- `span_start_offsets`/`span_end_offsets` (`i32` each, matching
//! [`lucene_search::term_vectors_query::TermOffsetSpan`]'s own field types
//! exactly, including its documented allowance for a negative or
//! out-of-range offset to simply be dropped rather than rejected -- see
//! `highlighter.rs`'s `assemble_fragments` doc comment), plus a caller-owned
//! term string per span via a concatenated `span_term_data` buffer sliced by
//! `span_term_lens` -- the same concatenated-buffer convention
//! `facets.rs`'s `ranges_from_raw` already uses for its per-range labels, for
//! the same reason (every term's length is already known to the caller up
//! front, so building one contiguous buffer is cheap on the C side).
//!
//! ## Output: a new [`crate::registry::FragmentResultsHandle`]
//!
//! Every assembled [`lucene_search::highlighter::Fragment`] carries a
//! variable-length `text` string *and* a variable-length `matched_terms`
//! list -- not a fixed-size element a parallel-buffer `_copy` call can
//! return in one shot (unlike `ffi_range_facet_counts`'s caller-allocated
//! `u64` buffer). Results are therefore collected into a new
//! [`crate::registry::FragmentResultsHandle`], read back via
//! `results_fragments.rs`'s accessors, exactly as `ffi_facet_counts_sorted_set`
//! does for its own variable-length `(ord, label, count)` output.

use std::os::raw::c_char;

use lucene_search::highlighter::{self, FragmentConfig};
use lucene_search::term_vectors_query::TermOffsetSpan;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;
use crate::registry::{fragment_results, lock_recovering, FragmentResultsHandle};

/// Parses `count` [`TermOffsetSpan`]s from four parallel input arrays -- see
/// this module's doc comment for the wire shape (`span_term_data`/
/// `span_term_lens` concatenated-buffer encoding, mirroring `facets.rs`'s
/// `ranges_from_raw`).
///
/// # Safety
/// `start_offsets`/`end_offsets` must be valid for reads of `count` `i32`s;
/// `term_lens` must be valid for reads of `count` `usize`s; `term_data` must
/// be valid for reads of `term_lens.iter().sum()` bytes (or null when that
/// sum is `0`).
unsafe fn spans_from_raw(
    term_data: *const u8,
    term_lens: *const usize,
    start_offsets: *const i32,
    end_offsets: *const i32,
    count: usize,
) -> Result<Vec<TermOffsetSpan>, FfiStatus> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if term_lens.is_null() || start_offsets.is_null() || end_offsets.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    // SAFETY: caller contract guarantees each of these is valid for `count`
    // reads of its element type.
    let lens = unsafe { std::slice::from_raw_parts(term_lens, count) };
    let starts = unsafe { std::slice::from_raw_parts(start_offsets, count) };
    let ends = unsafe { std::slice::from_raw_parts(end_offsets, count) };

    let total_term_len: usize = lens.iter().sum();
    let term_bytes: &[u8] = if total_term_len == 0 {
        &[]
    } else {
        if term_data.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `term_data` is valid for
        // `total_term_len` bytes whenever that sum is nonzero.
        unsafe { std::slice::from_raw_parts(term_data, total_term_len) }
    };

    let mut spans = Vec::with_capacity(count);
    let mut offset = 0usize;
    for i in 0..count {
        let len = lens[i];
        let term = std::str::from_utf8(&term_bytes[offset..offset + len])
            .map_err(|_| FfiStatus::InvalidUtf8)?
            .to_string();
        offset += len;
        spans.push(TermOffsetSpan {
            term,
            start_offset: starts[i],
            end_offset: ends[i],
        });
    }
    Ok(spans)
}

/// Assembles highlighted text fragments from `full_text` and `spans_count`
/// caller-supplied match spans -- wraps
/// `lucene_search::highlighter::assemble_fragments` with a
/// `lucene_search::highlighter::FragmentConfig` built from `window_chars`/
/// `pre`/`post`/`max_fragments`. Writes a new
/// [`crate::registry::FragmentResultsHandle`] to
/// `*out_fragment_results_handle` on success. `spans_count == 0` (or an empty
/// `full_text`) is not an error -- it simply produces zero fragments, per
/// `assemble_fragments`'s own documented "no matches" convention.
///
/// # Safety
/// `full_text` must be valid for `full_text_len` bytes (or null when
/// `full_text_len == 0`); `span_term_lens`/`span_start_offsets`/
/// `span_end_offsets` must each be valid for reads of `spans_count` of their
/// element type; `span_term_data` must be valid for reads of
/// `span_term_lens.iter().sum()` bytes (or null when that sum is `0`);
/// `pre`/`post` must be valid for `pre_len`/`post_len` bytes respectively (or
/// null when their length is `0`); `out_fragment_results_handle` must be
/// valid for one `u64` write. `snap_to_sentence` is a C `bool`-style flag
/// (`0` = fixed `window_chars` window, nonzero = sentence-boundary snapping)
/// -- see `lucene_search::highlighter`'s doc comment on sentence-boundary
/// snapping for the exact heuristic and its documented scope.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_assemble_fragments(
    full_text: *const c_char,
    full_text_len: usize,
    spans_count: usize,
    span_term_data: *const u8,
    span_term_lens: *const usize,
    span_start_offsets: *const i32,
    span_end_offsets: *const i32,
    window_chars: usize,
    pre: *const c_char,
    pre_len: usize,
    post: *const c_char,
    post_len: usize,
    max_fragments: usize,
    snap_to_sentence: u8,
    out_fragment_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_fragment_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `full_text` is valid for
        // `full_text_len` bytes and `pre`/`post` are valid for `pre_len`/
        // `post_len` bytes.
        let full_text = unsafe { str_from_raw(full_text as *const u8, full_text_len)? };
        let pre = unsafe { str_from_raw(pre as *const u8, pre_len)? };
        let post = unsafe { str_from_raw(post as *const u8, post_len)? };
        // SAFETY: caller contract guarantees the four span arrays are valid
        // per this function's `# Safety` section.
        let spans = unsafe {
            spans_from_raw(
                span_term_data,
                span_term_lens,
                span_start_offsets,
                span_end_offsets,
                spans_count,
            )?
        };

        if max_fragments == 0 {
            set_last_error("ffi_assemble_fragments: max_fragments must be greater than zero");
            return Err(FfiStatus::InvalidArgument);
        }

        let config = FragmentConfig {
            window_chars,
            pre: pre.to_string(),
            post: post.to_string(),
            max_fragments,
            snap_to_sentence: snap_to_sentence != 0,
        };

        let fragments = highlighter::assemble_fragments(full_text, &spans, &config);

        let handle =
            lock_recovering(fragment_results()).insert(FragmentResultsHandle { fragments });
        // SAFETY: caller contract guarantees `out_fragment_results_handle` is
        // valid for one write.
        unsafe {
            *out_fragment_results_handle = handle;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results_fragments::{
        ffi_close_fragment_results, ffi_fragment_result_matched_term,
        ffi_fragment_result_matched_terms_len, ffi_fragment_result_text, ffi_fragment_results_len,
    };

    fn span_arrays(spans: &[(&str, i32, i32)]) -> (Vec<u8>, Vec<usize>, Vec<i32>, Vec<i32>) {
        let mut term_data = Vec::new();
        let mut term_lens = Vec::new();
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for (term, start, end) in spans {
            term_data.extend_from_slice(term.as_bytes());
            term_lens.push(term.len());
            starts.push(*start);
            ends.push(*end);
        }
        (term_data, term_lens, starts, ends)
    }

    fn read_all_fragment_text(handle: u64, index: usize) -> String {
        let mut buf = [0 as c_char; 256];
        let mut written: usize = 0;
        let rc = unsafe {
            ffi_fragment_result_text(
                handle,
                index,
                buf.as_mut_ptr(),
                buf.len(),
                &mut written as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap()
            .to_string()
    }

    // Real-fixture-composed test, mirroring `highlighter.rs`'s own
    // `real_fixture_offsets_composed_with_their_real_field_text`: task
    // #3/#39's checked-in `fixtures/data/term_vectors_index/` fixture (doc
    // 0's "text" field, "cat"/"car"/"cat" at char offsets 0..3/4..7/8..11,
    // describing the literal text "cat car cat") composed with the real
    // text those offsets denote.
    #[test]
    fn assemble_fragments_matches_highlighter_rs_directly_on_real_fixture_offsets() {
        let full_text = "cat car cat";
        let spans = [("cat", 0i32, 3i32), ("car", 4, 7), ("cat", 8, 11)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let pre = "<b>";
        let post = "</b>";

        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                20,
                pre.as_ptr() as *const c_char,
                pre.len(),
                post.as_ptr() as *const c_char,
                post.len(),
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_fragment_results_len(out, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );

        // Cross-check against calling `lucene_search::highlighter` directly
        // with the same real fixture-derived spans, per the task brief's
        // differential requirement.
        let direct_spans = [
            TermOffsetSpan {
                term: "cat".to_string(),
                start_offset: 0,
                end_offset: 3,
            },
            TermOffsetSpan {
                term: "car".to_string(),
                start_offset: 4,
                end_offset: 7,
            },
            TermOffsetSpan {
                term: "cat".to_string(),
                start_offset: 8,
                end_offset: 11,
            },
        ];
        let expected = highlighter::assemble_fragments(
            full_text,
            &direct_spans,
            &FragmentConfig {
                window_chars: 20,
                pre: "<b>".to_string(),
                post: "</b>".to_string(),
                max_fragments: 5,
                snap_to_sentence: false,
            },
        );
        assert_eq!(len, expected.len());

        for (i, exp) in expected.iter().enumerate() {
            let text = read_all_fragment_text(out, i);
            assert_eq!(text, exp.text);

            let mut terms_len: usize = 0;
            assert_eq!(
                unsafe { ffi_fragment_result_matched_terms_len(out, i, &mut terms_len as *mut _) },
                FfiStatus::Ok.code()
            );
            assert_eq!(terms_len, exp.matched_terms.len());
            for (j, exp_term) in exp.matched_terms.iter().enumerate() {
                let mut buf = [0 as c_char; 64];
                let mut written: usize = 0;
                assert_eq!(
                    unsafe {
                        ffi_fragment_result_matched_term(
                            out,
                            i,
                            j,
                            buf.as_mut_ptr(),
                            buf.len(),
                            &mut written as *mut _,
                        )
                    },
                    FfiStatus::Ok.code()
                );
                let got = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                    .to_str()
                    .unwrap();
                assert_eq!(got, exp_term);
            }
        }

        // Single merged fragment expected (all three matches within a
        // 20-char window of each other), matching the direct-call test in
        // `highlighter.rs`.
        assert_eq!(len, 1);
        assert_eq!(
            read_all_fragment_text(out, 0),
            "<b>cat</b> <b>car</b> <b>cat</b>"
        );

        ffi_close_fragment_results(out);
    }

    #[test]
    fn assemble_fragments_empty_spans_yields_zero_fragments() {
        let full_text = "the quick brown fox";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_fragment_results_len(out, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 0);
        ffi_close_fragment_results(out);
    }

    #[test]
    fn assemble_fragments_empty_full_text_with_null_pointer_and_zero_len_is_ok() {
        let spans = [("cat", 0i32, 3i32)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                std::ptr::null(),
                0,
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_fragment_results_len(out, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 0);
        ffi_close_fragment_results(out);
    }

    #[test]
    fn assemble_fragments_out_of_range_spans_are_dropped_not_erroring() {
        let full_text = "cat dog";
        let spans = [("bad", 100i32, 200i32), ("cat", 0, 3)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_fragment_results_len(out, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        assert_eq!(len, 1);
        assert!(read_all_fragment_text(out, 0).contains("<b>cat</b>"));
        ffi_close_fragment_results(out);
    }

    #[test]
    fn assemble_fragments_null_out_handle_is_null_pointer_error() {
        let full_text = "cat dog";
        let spans = [("cat", 0i32, 3i32)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn assemble_fragments_null_full_text_with_nonzero_len_is_null_pointer_error() {
        let spans = [("cat", 0i32, 3i32)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                std::ptr::null(),
                5,
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn assemble_fragments_null_pre_with_nonzero_len_is_null_pointer_error() {
        let full_text = "cat dog";
        let spans = [("cat", 0i32, 3i32)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                std::ptr::null(),
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn assemble_fragments_null_post_with_nonzero_len_is_null_pointer_error() {
        let full_text = "cat dog";
        let spans = [("cat", 0i32, 3i32)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                std::ptr::null(),
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn assemble_fragments_null_span_term_lens_with_nonzero_spans_count_is_null_pointer_error() {
        let full_text = "cat dog";
        let starts = [0i32];
        let ends = [3i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn assemble_fragments_null_span_term_data_with_nonzero_term_len_is_null_pointer_error() {
        let full_text = "cat dog";
        let term_lens = [3usize];
        let starts = [0i32];
        let ends = [3i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                1,
                std::ptr::null(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn assemble_fragments_invalid_utf8_term_is_invalid_utf8_error() {
        let full_text = "cat dog";
        let term_data = [0xFFu8];
        let term_lens = [1usize];
        let starts = [0i32];
        let ends = [3i32];
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                1,
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidUtf8.code());
    }

    #[test]
    fn assemble_fragments_zero_max_fragments_is_invalid_argument() {
        let full_text = "cat dog";
        let spans = [("cat", 0i32, 3i32)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                0,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
    }

    #[test]
    fn assemble_fragments_with_all_empty_term_spans_is_ok() {
        // spans_count > 0 but every term is an empty string, so
        // `total_term_len == 0` for a reason other than `count == 0` --
        // `span_term_data` may legitimately be null here.
        let full_text = "cat dog";
        let spans = [("", 0i32, 0i32), ("", 4i32, 4i32)];
        let (_, term_lens, starts, ends) = span_arrays(&spans);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                spans.len(),
                std::ptr::null(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                10,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_fragment_results(out);
    }

    #[test]
    fn assemble_fragments_snap_to_sentence_flag_matches_direct_call() {
        // Cross-checks the `snap_to_sentence` wire flag against calling
        // `lucene_search::highlighter::assemble_fragments` directly with
        // `snap_to_sentence: true` -- same real-world shape as
        // `highlighter.rs`'s own `sentence_snap_changes_output_vs_naive_fixed_window`
        // test, ported across the FFI boundary.
        let full_text = "Cats are great pets. Dogs are loyal companions too.";
        let start = full_text.find("great").unwrap() as i32;
        let end = start + "great".len() as i32;
        let spans = [("great", start, end)];
        let (term_data, term_lens, starts, ends) = span_arrays(&spans);
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_assemble_fragments(
                full_text.as_ptr() as *const c_char,
                full_text.len(),
                spans.len(),
                term_data.as_ptr(),
                term_lens.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                15,
                "<b>".as_ptr() as *const c_char,
                3,
                "</b>".as_ptr() as *const c_char,
                4,
                5,
                1,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let text = read_all_fragment_text(out, 0);
        let expected = highlighter::assemble_fragments(
            full_text,
            &[TermOffsetSpan {
                term: "great".to_string(),
                start_offset: start,
                end_offset: end,
            }],
            &FragmentConfig {
                window_chars: 15,
                pre: "<b>".to_string(),
                post: "</b>".to_string(),
                max_fragments: 5,
                snap_to_sentence: true,
            },
        );
        assert_eq!(expected.len(), 1);
        assert_eq!(text, expected[0].text);
        assert!(text.starts_with("Cats are"));
        assert!(!text.contains("Dogs"));

        ffi_close_fragment_results(out);
    }
}
