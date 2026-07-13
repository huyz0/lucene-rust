//! `ffi_search_term_query`/`ffi_search_boolean_query`/`ffi_search_phrase_query`:
//! runs this port's existing `lucene_search::search_*_query` functions
//! against an already-opened [`crate::segment::SegmentHandle`], collecting
//! every matching, live doc ID into a new [`crate::registry::ResultsHandle`]
//! entirely Rust-side (a plain [`lucene_search::VecCollector`] -- no
//! callback ever crosses back into the caller, per the `ffi-safety` skill).
//!
//! `live_docs` is always `None` here (no `.liv`/deletions FFI surface yet --
//! deferred, tracked in `docs/parity.md`): every matched doc is reported.
//!
//! **Nested `BooleanQuery` clauses (task #25)**: `lucene_search::BooleanQuery`'s
//! `must`/`should`/`must_not` are now `Vec<Clause>` (a `Clause` is a `TermQuery` or
//! a nested `BooleanQuery`, recursively -- see that crate's `query` module doc).
//! This FFI surface deliberately keeps constructing only flat `Clause::Term`
//! clauses from its four-parallel-array wire format -- `read_term_clauses` builds
//! a `Vec<TermQuery>`, and `BooleanQuery::with_must`/`with_should`/`with_must_not`
//! accept it unchanged (each `TermQuery` converts to `Clause::Term` via `Clause`'s
//! `From<TermQuery>` impl), so no FFI-side code change was needed for this crate to
//! keep compiling. Exposing nested-boolean *construction* over the C ABI (a
//! `BooleanQuery` clause list containing another whole clause list, `Occur`-tagged)
//! is a real wire-format design question of its own -- deferred here as a
//! documented decision, not an oversight, since this task's scope is
//! `lucene-search`'s own nested-clause support, not a new FFI capability.

use std::os::raw::c_char;

use lucene_codecs::postings::{DocInput, PosInput};
use lucene_search::{search_boolean_query, search_phrase_query, search_term_query};
use lucene_search::{BooleanQuery, PhraseQuery, TermQuery, VecCollector};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{lock_recovering, results, segments, ResultsHandle};

fn map_search_error(e: lucene_search::Error) -> FfiStatus {
    set_last_error(format!("search failed: {e}"));
    FfiStatus::Search
}

/// Test-only panic-injection switch for
/// `registry_mutex_recovers_from_poisoning_after_a_panic_mid_query` below:
/// there is no way to reach a real internal `unwrap()`/indexing panic from
/// adversarial-but-otherwise-well-formed bytes through this crate's public
/// decode paths (every decoder here already turns corrupted bytes into an
/// `Err` -> `FfiStatus::Decode`, per `segment.rs`'s garbage-bytes tests), so
/// this flag fabricates the one thing a real panic there would have in
/// common with any other panic: it fires *while `ffi_search_term_query`
/// still holds the `segments()` registry's `MutexGuard`*, the exact
/// condition that poisons the mutex. Never armed outside a test, and always
/// disarmed (via `swap`) the instant it fires, so it can't leak into any
/// other test.
#[cfg(test)]
static PANIC_ON_NEXT_TERM_QUERY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn arm_panic_on_next_term_query() {
    PANIC_ON_NEXT_TERM_QUERY.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Runs `search_term_query` for `(field, term)` against `segment_handle`,
/// writing a new results handle to `*out_results_handle` on success.
///
/// # Safety
/// `field` must be valid for `field_len` bytes, `term` for `term_len`
/// bytes, `out_results_handle` valid for one `u64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_search_term_query(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    term: *const u8,
    term_len: usize,
    out_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field`/`term` are valid for their
        // paired lengths.
        let (field, term) = unsafe {
            (
                str_from_raw(field as *const u8, field_len)?,
                bytes_from_raw(term, term_len)?,
            )
        };
        let query = TermQuery::new(field, term.to_vec());

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_search_term_query: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;

        // Test-only: see `arm_panic_on_next_term_query`'s doc comment. Fires
        // while `segments` (the `MutexGuard` above) is still held, exactly
        // like a real decode panic reached through `search_term_query` below
        // would.
        #[cfg(test)]
        if PANIC_ON_NEXT_TERM_QUERY.swap(false, std::sync::atomic::Ordering::SeqCst) {
            panic!("test-only simulated panic while the segments registry lock is held");
        }

        let doc_in = segment
            .doc_bytes
            .as_deref()
            .map(|b| DocInput::open(b, &segment.segment_id, &segment.segment_suffix))
            .transpose()
            .map_err(|e| {
                set_last_error(format!("reopening .doc: {e}"));
                FfiStatus::Decode
            })?;

        let mut collector = VecCollector::default();
        search_term_query(
            &segment.fields,
            doc_in.as_ref(),
            None,
            &query,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = lock_recovering(results()).insert(ResultsHandle {
            docs: collector.docs,
        });
        // SAFETY: caller contract guarantees `out_results_handle` is valid for one write.
        unsafe {
            *out_results_handle = handle;
        }
        Ok(())
    })
}

/// Reads `count` `(field, term)` pairs from four parallel flat arrays into
/// a `Vec<TermQuery>` -- the shared clause-array decoder for
/// [`ffi_search_boolean_query`]'s `must`/`should`/`must_not` clause lists.
/// `count == 0` is valid or `fields`/`field_lens`/`terms`/`term_lens` are
/// null (and never dereferenced in that case).
unsafe fn read_term_clauses(
    fields: *const *const c_char,
    field_lens: *const usize,
    terms: *const *const u8,
    term_lens: *const usize,
    count: usize,
) -> Result<Vec<TermQuery>, FfiStatus> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if fields.is_null() || field_lens.is_null() || terms.is_null() || term_lens.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: caller contract guarantees each array is valid for `count`
        // elements, and each element pair is valid for its paired length.
        let (field, term) = unsafe {
            let field_ptr = *fields.add(i);
            let field_len = *field_lens.add(i);
            let term_ptr = *terms.add(i);
            let term_len = *term_lens.add(i);
            (
                str_from_raw(field_ptr as *const u8, field_len)?,
                bytes_from_raw(term_ptr, term_len)?,
            )
        };
        out.push(TermQuery::new(field, term.to_vec()));
    }
    Ok(out)
}

/// Runs `search_boolean_query` against `segment_handle`. Each of
/// `must`/`should`/`must_not` is passed as four parallel flat arrays
/// (`*_fields`, `*_field_lens`, `*_terms`, `*_term_lens`) of length
/// `*_count` -- element `i` is one `TermQuery` clause `(fields[i][..field_lens[i]],
/// terms[i][..term_lens[i]])`. An empty clause bucket is `count == 0` with
/// its four array pointers allowed to be null.
///
/// # Safety
/// Every `(pointer, len)` / `(array, count)` pair must be valid for the
/// documented reads; `out_results_handle` must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_boolean_query(
    segment_handle: u64,
    must_fields: *const *const c_char,
    must_field_lens: *const usize,
    must_terms: *const *const u8,
    must_term_lens: *const usize,
    must_count: usize,
    should_fields: *const *const c_char,
    should_field_lens: *const usize,
    should_terms: *const *const u8,
    should_term_lens: *const usize,
    should_count: usize,
    must_not_fields: *const *const c_char,
    must_not_field_lens: *const usize,
    must_not_terms: *const *const u8,
    must_not_term_lens: *const usize,
    must_not_count: usize,
    out_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: see `read_term_clauses`'s contract; every array/count pair here
        // matches it exactly.
        let query = unsafe {
            BooleanQuery::new()
                .with_must(read_term_clauses(
                    must_fields,
                    must_field_lens,
                    must_terms,
                    must_term_lens,
                    must_count,
                )?)
                .with_should(read_term_clauses(
                    should_fields,
                    should_field_lens,
                    should_terms,
                    should_term_lens,
                    should_count,
                )?)
                .with_must_not(read_term_clauses(
                    must_not_fields,
                    must_not_field_lens,
                    must_not_terms,
                    must_not_term_lens,
                    must_not_count,
                )?)
        };

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_search_boolean_query: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;
        let doc_in = segment
            .doc_bytes
            .as_deref()
            .map(|b| DocInput::open(b, &segment.segment_id, &segment.segment_suffix))
            .transpose()
            .map_err(|e| {
                set_last_error(format!("reopening .doc: {e}"));
                FfiStatus::Decode
            })?;

        let mut collector = VecCollector::default();
        search_boolean_query(
            &segment.fields,
            doc_in.as_ref(),
            None,
            &query,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = lock_recovering(results()).insert(ResultsHandle {
            docs: collector.docs,
        });
        // SAFETY: caller contract guarantees `out_results_handle` is valid for one write.
        unsafe {
            *out_results_handle = handle;
        }
        Ok(())
    })
}

/// Runs `search_phrase_query` for `field`'s `term_count`-term phrase
/// (`terms[i]`/`term_lens[i]`, in phrase order) against `segment_handle`.
/// A single-term phrase never needs the segment's `.pos` file (delegates to
/// `search_term_query`, see [`lucene_search::search_phrase_query`]'s doc
/// comment); a multi-term phrase requires the segment to have been opened
/// with a `.pos` file ([`crate::segment::ffi_open_segment`]'s `pos_name`
/// parameter) -- otherwise this returns [`FfiStatus::Search`].
///
/// # Safety
/// `field` must be valid for `field_len` bytes; `terms`/`term_lens` must
/// each be valid for `term_count` elements, with every `terms[i]` valid for
/// `term_lens[i]` bytes; `out_results_handle` must be valid for one `u64`
/// write.
#[no_mangle]
pub unsafe extern "C" fn ffi_search_phrase_query(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    terms: *const *const u8,
    term_lens: *const usize,
    term_count: usize,
    out_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field` is valid for `field_len`
        // bytes, and (when `term_count > 0`) `terms`/`term_lens` are valid for
        // `term_count` elements with each element pair valid for its length.
        let (field, term_list) = unsafe {
            let field = str_from_raw(field as *const u8, field_len)?;
            let mut term_list = Vec::with_capacity(term_count);
            if term_count > 0 {
                if terms.is_null() || term_lens.is_null() {
                    return Err(FfiStatus::NullPointer);
                }
                for i in 0..term_count {
                    let term_ptr = *terms.add(i);
                    let term_len = *term_lens.add(i);
                    term_list.push(bytes_from_raw(term_ptr, term_len)?.to_vec());
                }
            }
            (field, term_list)
        };
        let query = PhraseQuery::new(field, term_list);

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_search_phrase_query: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;
        let doc_in = segment
            .doc_bytes
            .as_deref()
            .map(|b| DocInput::open(b, &segment.segment_id, &segment.segment_suffix))
            .transpose()
            .map_err(|e| {
                set_last_error(format!("reopening .doc: {e}"));
                FfiStatus::Decode
            })?;
        let pos_in = segment
            .pos_bytes
            .as_deref()
            .map(|b| PosInput::open(b, &segment.segment_id, &segment.segment_suffix))
            .transpose()
            .map_err(|e| {
                set_last_error(format!("reopening .pos: {e}"));
                FfiStatus::Decode
            })?;

        let mut collector = VecCollector::default();
        search_phrase_query(
            &segment.fields,
            doc_in.as_ref(),
            pos_in.as_ref(),
            None,
            None,
            &query,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = lock_recovering(results()).insert(ResultsHandle {
            docs: collector.docs,
        });
        // SAFETY: caller contract guarantees `out_results_handle` is valid for one write.
        unsafe {
            *out_results_handle = handle;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};
    use crate::results::{ffi_close_results, ffi_results_copy, ffi_results_len};
    use crate::segment::{ffi_close_segment, ffi_open_segment};

    fn fixture_dir_path() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        )
        .to_string()
    }

    fn segment_id_bytes() -> [u8; 16] {
        let hex = "a461ec6668896df01024ada528579052";
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    fn open_dir() -> u64 {
        let path = fixture_dir_path();
        let mut handle: u64 = 0;
        unsafe {
            ffi_open_directory(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            );
        }
        handle
    }

    fn open_segment(dir_handle: u64, with_pos: bool) -> u64 {
        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let doc = "_0_Lucene104_0.doc";
        let pos = "_0_Lucene104_0.pos";
        let suffix = "Lucene104_0";
        let id = segment_id_bytes();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr() as *const c_char,
                fnm.len(),
                tim.as_ptr() as *const c_char,
                tim.len(),
                tip.as_ptr() as *const c_char,
                tip.len(),
                tmd.as_ptr() as *const c_char,
                tmd.len(),
                doc.as_ptr() as *const c_char,
                doc.len(),
                if with_pos {
                    pos.as_ptr() as *const c_char
                } else {
                    std::ptr::null()
                },
                if with_pos { pos.len() } else { 0 },
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8957,
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    fn read_results(results_handle: u64) -> Vec<i32> {
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_results_len(results_handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        let mut buf = vec![0i32; len];
        assert_eq!(
            unsafe { ffi_results_copy(results_handle, buf.as_mut_ptr(), buf.len()) },
            FfiStatus::Ok.code()
        );
        buf
    }

    #[test]
    fn term_query_body_cat_returns_expected_docs() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_results(results_handle), vec![0, 2]);

        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_id_field_needs_no_doc_file() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "id";
        let term = b"id2";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_results(results_handle), vec![2]);

        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_missing_term_returns_empty_results_not_an_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"zzz-missing";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_results(results_handle).is_empty());

        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_unknown_segment_handle_is_invalid_handle() {
        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    /// A directory handle passed where a segment handle is expected must be
    /// rejected by the registry-tag check on `segments().get(...)`, not
    /// accidentally treated as a (coincidentally same-bit-pattern) segment
    /// handle -- see `handle.rs`'s `RegistryTag`.
    #[test]
    fn term_query_directory_handle_passed_as_segment_handle_is_invalid_handle() {
        let dir_handle = open_dir();
        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                dir_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_after_segment_closed_is_invalid_handle_not_a_crash() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        assert_eq!(ffi_close_segment(seg_handle), FfiStatus::Ok.code());

        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let term = b"cat";
        let rc = unsafe {
            ffi_search_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_must_cat_must_not_bird_matches_expected_doc() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let must_field = "body";
        let must_term = b"cat";
        let must_not_field = "body";
        let must_not_term = b"bird";

        let must_fields = [must_field.as_ptr() as *const c_char];
        let must_field_lens = [must_field.len()];
        let must_terms = [must_term.as_ptr()];
        let must_term_lens = [must_term.len()];

        let must_not_fields = [must_not_field.as_ptr() as *const c_char];
        let must_not_field_lens = [must_not_field.len()];
        let must_not_terms = [must_not_term.as_ptr()];
        let must_not_term_lens = [must_not_term.len()];

        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query(
                seg_handle,
                must_fields.as_ptr(),
                must_field_lens.as_ptr(),
                must_terms.as_ptr(),
                must_term_lens.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                must_not_fields.as_ptr(),
                must_not_field_lens.as_ptr(),
                must_not_terms.as_ptr(),
                must_not_term_lens.as_ptr(),
                1,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        // body/cat -> [0, 2]; body/bird -> [1, 4]; must_not removes none of them.
        assert_eq!(read_results(results_handle), vec![0, 2]);

        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_no_clauses_matches_nothing() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_results(results_handle).is_empty());
        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_unknown_segment_handle_is_invalid_handle() {
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query(
                0xFFFF,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn phrase_query_single_term_delegates_to_term_query() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat".as_slice();
        let terms = [term.as_ptr()];
        let term_lens = [term.len()];

        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                terms.as_ptr(),
                term_lens.as_ptr(),
                1,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_results(results_handle), vec![0, 2]);

        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_multi_term_without_pos_file_is_search_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false); // no .pos opened

        let field = "body";
        let t1 = b"cat".as_slice();
        let t2 = b"dog".as_slice();
        let terms = [t1.as_ptr(), t2.as_ptr()];
        let term_lens = [t1.len(), t2.len()];

        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                terms.as_ptr(),
                term_lens.as_ptr(),
                2,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Search.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_empty_terms_matches_nothing() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_results(results_handle).is_empty());
        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Swaps a live segment handle's `.doc` bytes for garbage that fails
    /// `DocInput::open`'s header check, so the "reopen the `.doc` file for
    /// this query" `map_err` branch in `ffi_search_term_query`/
    /// `ffi_search_boolean_query`/`ffi_search_phrase_query` is reachable --
    /// this can't happen through the public API alone since `ffi_open_segment`
    /// already validates the `.doc` bytes once at open time.
    fn corrupt_doc_bytes(seg_handle: u64) {
        let mut segments = lock_recovering(segments());
        let segment = segments.get_mut(seg_handle).expect("segment handle");
        segment.doc_bytes = Some(vec![0u8; 4]);
    }

    /// Same idea as [`corrupt_doc_bytes`], for the `.pos` file.
    fn corrupt_pos_bytes(seg_handle: u64) {
        let mut segments = lock_recovering(segments());
        let segment = segments.get_mut(seg_handle).expect("segment handle");
        segment.pos_bytes = Some(vec![0u8; 4]);
    }

    #[test]
    fn term_query_doc_reopen_failure_is_decode_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        corrupt_doc_bytes(seg_handle);

        let field = "body";
        let term = b"cat"; // docFreq == 2, needs the (now-corrupted) .doc file.
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let rc = unsafe {
            ffi_search_boolean_query(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_must_clause_with_null_arrays_and_nonzero_count_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                1, // count > 0 but every array pointer is null.
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_should_clause_with_null_arrays_and_nonzero_count_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                1, // count > 0 but every array pointer is null.
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_must_not_clause_with_null_arrays_and_nonzero_count_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                1, // count > 0 but every array pointer is null.
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_doc_reopen_failure_is_decode_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        corrupt_doc_bytes(seg_handle);

        let must_field = "body";
        let must_term = b"cat";
        let must_fields = [must_field.as_ptr() as *const c_char];
        let must_field_lens = [must_field.len()];
        let must_terms = [must_term.as_ptr()];
        let must_term_lens = [must_term.len()];

        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query(
                seg_handle,
                must_fields.as_ptr(),
                must_field_lens.as_ptr(),
                must_terms.as_ptr(),
                must_term_lens.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let rc = unsafe {
            ffi_search_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_nonzero_term_count_with_null_arrays_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                std::ptr::null(),
                2,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_unknown_segment_handle_is_invalid_handle() {
        let field = "body";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn phrase_query_doc_reopen_failure_is_decode_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        corrupt_doc_bytes(seg_handle);

        let field = "body";
        let term = b"cat".as_slice();
        let terms = [term.as_ptr()];
        let term_lens = [term.len()];
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                terms.as_ptr(),
                term_lens.as_ptr(),
                1,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_pos_reopen_failure_is_decode_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, true);
        corrupt_pos_bytes(seg_handle);

        let field = "body";
        let t1 = b"cat".as_slice();
        let t2 = b"dog".as_slice();
        let terms = [t1.as_ptr(), t2.as_ptr()];
        let term_lens = [t1.len(), t2.len()];
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                terms.as_ptr(),
                term_lens.as_ptr(),
                2,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Decode.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Regression test for the mutex-poisoning fix in `registry::lock_recovering`:
    /// a panic while `ffi_search_term_query` holds the `segments()` registry's
    /// lock must be caught by `guard` (reported as `FfiStatus::Panic`, not a
    /// crash) *and* must not permanently wedge that registry -- a later,
    /// unrelated, well-formed call against the same segment handle must still
    /// succeed. Before the fix, the second call would itself panic (a poisoned
    /// `Mutex::lock().unwrap()` panics) and also return `FfiStatus::Panic`,
    /// forever.
    #[test]
    fn registry_mutex_recovers_from_poisoning_after_a_panic_mid_query() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";

        arm_panic_on_next_term_query();
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(
            rc,
            FfiStatus::Panic.code(),
            "the injected panic must be caught by `guard`, not crash the process"
        );
        assert!(segments().is_poisoned(), "the panic must poison the mutex");

        // A subsequent, unrelated, well-formed call against the *same*
        // registry (and the same still-live segment handle) must succeed --
        // proving `lock_recovering` recovered the poisoned mutex rather than
        // leaving every future call on this registry permanently broken.
        let rc = unsafe {
            ffi_search_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_results(results_handle), vec![0, 2]);

        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }
}
