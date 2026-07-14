//! `ffi_open_directory_reader`/`ffi_close_directory_reader` plus
//! `ffi_search_term_query_multi_segment`/`ffi_search_boolean_query_multi_segment`
//! (task #51): the FFI exposure of task #45's
//! `lucene_search::directory_reader::DirectoryReader` (open every segment a
//! commit lists, in one call) and task #41's
//! `search_term_query_multi_segment`/`search_boolean_query_multi_segment`
//! (fan out a query across every open segment and merge into one globally-
//! ranked result) -- the last "lucene-ffi exposure" gap those two tasks'
//! own doc comments flagged as deferred (see `lucene-search/src/directory_reader.rs`'s
//! and `multi_segment.rs`'s module docs, and `docs/parity.md`).
//!
//! **Why a path string, not an already-open [`crate::directory::ffi_open_directory`]
//! handle**: unlike [`crate::segment::ffi_open_segment`] (which composes with an
//! already-open directory handle because a caller may want to open several
//! segments from the same directory with different file-name combinations),
//! opening a `DirectoryReader` is a single, self-contained operation --
//! `DirectoryReader::open` needs a `&dyn Directory` only for the duration of
//! the open call itself (every segment's bytes are copied into the resulting
//! `DirectoryReader`'s own owned buffers, see `directory_reader.rs`'s
//! `SegmentReader` doc comment), so there is no reason to keep a separate
//! directory handle alive afterwards. This entry point therefore takes the
//! path directly (the same `(*const c_char, len)` UTF-8 convention
//! [`crate::directory::ffi_open_directory`] already established) and opens
//! its own internal, short-lived `FsDirectory` rather than requiring the
//! caller to separately open and manage one.
//!
//! **Results-handle reuse**: both multi-segment entry points below write
//! their `(doc_id, score)` hits into a [`crate::registry::ScoredResultsHandle`]
//! -- the exact same handle/registry task #30's single-segment
//! `ffi_search_term_query_scored`/`ffi_search_boolean_query_scored` already
//! use, read back via the same existing
//! [`crate::results_scored::ffi_scored_results_len`]/[`crate::results_scored::ffi_scored_results_copy`]/
//! [`crate::results_scored::ffi_close_scored_results`] trio. Multi-segment
//! search returns exactly the same `Vec<ScoreDoc>` shape single-segment
//! scored search already returns (see `multi_segment.rs`'s own module doc:
//! the merge step's output is indistinguishable in shape from any single
//! collector's `top_docs()`), so a new results type/registry would be a
//! pointless duplicate of an already-shipped wire contract -- exactly the
//! non-duplication reasoning `registry.rs`'s own handle doc comments already
//! use to justify *not* merging `ScoredResultsHandle` into `ResultsHandle`
//! (a genuine shape difference) while justifying *not* inventing a third
//! shape here (no shape difference at all).
//!
//! **No norms**: task #45's `DirectoryReader` carries no `.nvm`/`.nvd` data
//! per segment (see that module's doc comment), so every per-segment norms
//! slot passed to `search_term_query_multi_segment`/
//! `search_boolean_query_multi_segment` here is `None` -- the same
//! documented `UNNORMED_FIELD_LENGTH` fallback this crate's single-segment
//! scored queries already use for a bare `None`, not a new approximation
//! introduced by this module.

use std::os::raw::c_char;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::{
    search_boolean_query_multi_segment, search_boolean_query_multi_segment_concurrent,
    search_term_query_multi_segment, search_term_query_multi_segment_concurrent,
};
use lucene_search::{BooleanQuery, TermQuery};
use lucene_store::directory::FsDirectory;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::query::{map_search_error, read_term_clauses};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{
    directory_readers, lock_recovering, scored_results, DirectoryReaderHandle, ScoredResultsHandle,
};

fn map_open_error(e: lucene_search::directory_reader::Error) -> FfiStatus {
    set_last_error(format!("opening directory reader: {e}"));
    match &e {
        lucene_search::directory_reader::Error::Store(lucene_store::Error::Io(_)) => FfiStatus::Io,
        _ => FfiStatus::Decode,
    }
}

// Test-only panic-injection switch for
// `registry_mutex_recovers_from_poisoning_after_a_panic_mid_multi_segment_query`
// below -- same purpose/shape as `query.rs`'s
// `PANIC_ON_NEXT_SCORED_TERM_QUERY` (see that flag's doc comment for the
// full rationale): **thread-local, not a process-wide `AtomicBool`**,
// specifically to avoid the flaky-test failure mode task #29's history
// documents (a shared atomic armed by one test firing inside a *different*,
// concurrently-running test's call to the same function). Arming and firing
// both happen on the same test's own thread, so no other test's thread can
// ever observe or trigger it.
#[cfg(test)]
thread_local! {
    static PANIC_ON_NEXT_MULTI_SEGMENT_QUERY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn arm_panic_on_next_multi_segment_query() {
    PANIC_ON_NEXT_MULTI_SEGMENT_QUERY.with(|c| c.set(true));
}

/// Opens every segment listed in the latest commit found at the `path_len`-byte
/// UTF-8 path `path` (`DirectoryReader.open(Directory)`-equivalent, see
/// `lucene_search::directory_reader::DirectoryReader::open`), writing the new
/// reader handle to `*out_handle` on success.
///
/// # Safety
/// `path` must be valid for reads of `path_len` bytes; `out_handle` must be
/// valid for one `u64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_open_directory_reader(
    path: *const c_char,
    path_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `path` is valid for `path_len` bytes.
        let path_str = unsafe { str_from_raw(path as *const u8, path_len) }?;
        let dir = FsDirectory::open(path_str);
        let reader = DirectoryReader::open(&dir).map_err(map_open_error)?;

        let handle = lock_recovering(directory_readers()).insert(DirectoryReaderHandle { reader });
        // SAFETY: caller contract guarantees `out_handle` is valid for one write.
        unsafe {
            *out_handle = handle;
        }
        Ok(())
    })
}

/// Closes a directory-reader handle opened by [`ffi_open_directory_reader`].
/// Returns [`FfiStatus::InvalidHandle`] for an unknown/already-closed handle.
#[no_mangle]
pub extern "C" fn ffi_close_directory_reader(handle: u64) -> i32 {
    guard(|| {
        lock_recovering(directory_readers())
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                set_last_error("ffi_close_directory_reader: unknown or already-closed handle");
                FfiStatus::InvalidHandle
            })
    })
}

/// Multi-segment sibling of [`crate::query::ffi_search_term_query_scored`]: runs
/// `search_term_query_multi_segment` for `(field, term)` against every segment
/// `reader_handle` has open, keeping the best `top_n` globally-ranked
/// `(doc_id, score)` hits in a new [`ScoredResultsHandle`] (see this module's
/// doc comment for why the same handle type task #30 already shipped).
///
/// # Safety
/// `field` must be valid for `field_len` bytes, `term` for `term_len` bytes,
/// `out_scored_results_handle` valid for one `u64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_search_term_query_multi_segment(
    reader_handle: u64,
    field: *const c_char,
    field_len: usize,
    term: *const u8,
    term_len: usize,
    top_n: usize,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
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

        let readers = lock_recovering(directory_readers());
        let reader_handle_value = readers.get(reader_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_term_query_multi_segment: unknown or already-closed reader handle",
            );
            FfiStatus::InvalidHandle
        })?;

        // Test-only: see `arm_panic_on_next_multi_segment_query`'s doc comment.
        // Fires while `readers` (the `MutexGuard` above) is still held, exactly
        // like a real decode panic reached through the search call below would.
        #[cfg(test)]
        if PANIC_ON_NEXT_MULTI_SEGMENT_QUERY.with(|c| c.replace(false)) {
            panic!("test-only simulated panic while the directory_readers registry lock is held");
        }

        let opened = reader_handle_value.reader.open_segments().map_err(|e| {
            set_last_error(format!("reopening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        let segments = opened.as_open_segments();
        let norms: Vec<Option<&FieldNorms<'_>>> = vec![None; segments.len()];

        let hits = search_term_query_multi_segment(&segments, &query, &norms, top_n)
            .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle { hits });
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// Multi-segment sibling of [`crate::query::ffi_search_boolean_query_scored`]: same
/// flat, `Clause::Term`-only four-parallel-array clause wire format (see
/// `query.rs`'s module doc), run against every segment `reader_handle` has
/// open via `search_boolean_query_multi_segment`, keeping the best `top_n`
/// globally-ranked `(doc_id, score)` hits in a new [`ScoredResultsHandle`].
///
/// # Safety
/// Every `(pointer, len)` / `(array, count)` pair must be valid for the
/// documented reads (see [`crate::query::ffi_search_boolean_query`]'s `# Safety`
/// section, which this mirrors exactly); `out_scored_results_handle` must be
/// valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_boolean_query_multi_segment(
    reader_handle: u64,
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
    top_n: usize,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
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

        let readers = lock_recovering(directory_readers());
        let reader_handle_value = readers.get(reader_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_boolean_query_multi_segment: unknown or already-closed reader handle",
            );
            FfiStatus::InvalidHandle
        })?;

        let opened = reader_handle_value.reader.open_segments().map_err(|e| {
            set_last_error(format!("reopening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        let segments = opened.as_open_segments();
        let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
            vec![None; segments.len()];

        let hits = search_boolean_query_multi_segment(&segments, &query, &norms, top_n)
            .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle { hits });
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// Concurrent sibling of [`ffi_search_term_query_multi_segment`]: identical
/// wire format, handle validation, and results-handle plumbing, but fans the
/// per-segment search out across `rayon`'s global pool via
/// `search_term_query_multi_segment_concurrent` instead of running segments
/// one at a time. See that function's doc comment (`multi_segment.rs`) for
/// why this is provably byte-for-byte identical to the sequential path for
/// the same input, not merely usually-the-same -- also exercised directly by
/// this module's own
/// `term_query_multi_segment_concurrent_matches_sequential_ffi_wrapper` test.
///
/// # Safety
/// Same contract as [`ffi_search_term_query_multi_segment`]: `field` must be
/// valid for `field_len` bytes, `term` for `term_len` bytes,
/// `out_scored_results_handle` valid for one `u64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_search_term_query_multi_segment_concurrent(
    reader_handle: u64,
    field: *const c_char,
    field_len: usize,
    term: *const u8,
    term_len: usize,
    top_n: usize,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
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

        let readers = lock_recovering(directory_readers());
        let reader_handle_value = readers.get(reader_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_term_query_multi_segment_concurrent: unknown or already-closed reader handle",
            );
            FfiStatus::InvalidHandle
        })?;

        let opened = reader_handle_value.reader.open_segments().map_err(|e| {
            set_last_error(format!("reopening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        let segments = opened.as_open_segments();
        let norms: Vec<Option<&FieldNorms<'_>>> = vec![None; segments.len()];

        let hits = search_term_query_multi_segment_concurrent(&segments, &query, &norms, top_n)
            .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle { hits });
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// Concurrent sibling of [`ffi_search_boolean_query_multi_segment`]: identical
/// wire format, handle validation, and results-handle plumbing, but fans the
/// per-segment search out via `search_boolean_query_multi_segment_concurrent`
/// instead of running segments one at a time. See
/// [`ffi_search_term_query_multi_segment_concurrent`]'s doc comment for the
/// same identical-output rationale, applied here to the boolean-query case.
///
/// # Safety
/// Same contract as [`ffi_search_boolean_query_multi_segment`]: every
/// `(pointer, len)` / `(array, count)` pair must be valid for the documented
/// reads (see [`crate::query::ffi_search_boolean_query`]'s `# Safety`
/// section, which this mirrors exactly); `out_scored_results_handle` must be
/// valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_boolean_query_multi_segment_concurrent(
    reader_handle: u64,
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
    top_n: usize,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
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

        let readers = lock_recovering(directory_readers());
        let reader_handle_value = readers.get(reader_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_boolean_query_multi_segment_concurrent: unknown or already-closed reader handle",
            );
            FfiStatus::InvalidHandle
        })?;

        let opened = reader_handle_value.reader.open_segments().map_err(|e| {
            set_last_error(format!("reopening segment postings: {e}"));
            FfiStatus::Decode
        })?;
        let segments = opened.as_open_segments();
        let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
            vec![None; segments.len()];

        let hits = search_boolean_query_multi_segment_concurrent(&segments, &query, &norms, top_n)
            .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle { hits });
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results_scored::{
        ffi_close_scored_results, ffi_scored_results_copy, ffi_scored_results_len,
    };

    fn fixture_dir_path() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        )
        .to_string()
    }

    fn open_reader() -> u64 {
        let path = fixture_dir_path();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_directory_reader(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_ne!(handle, 0);
        handle
    }

    fn read_scored_results(handle: u64) -> Vec<(i32, f32)> {
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_scored_results_len(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        let mut doc_ids = vec![0i32; len];
        let mut scores = vec![0.0f32; len];
        assert_eq!(
            unsafe {
                ffi_scored_results_copy(handle, doc_ids.as_mut_ptr(), scores.as_mut_ptr(), len)
            },
            FfiStatus::Ok.code()
        );
        doc_ids.into_iter().zip(scores).collect()
    }

    #[test]
    fn open_and_close_directory_reader_roundtrips() {
        let handle = open_reader();
        assert!(lock_recovering(directory_readers()).get(handle).is_some());
        assert_eq!(ffi_close_directory_reader(handle), FfiStatus::Ok.code());
        assert!(lock_recovering(directory_readers()).get(handle).is_none());
    }

    #[test]
    fn open_directory_reader_missing_commit_is_io_or_decode_error() {
        let path = std::env::temp_dir()
            .join(format!(
                "lucene-ffi-directory-reader-missing-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ))
            .to_string_lossy()
            .into_owned();
        std::fs::create_dir_all(&path).unwrap();
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_directory_reader(
                path.as_ptr() as *const c_char,
                path.len(),
                &mut handle as *mut _,
            )
        };
        assert_ne!(rc, FfiStatus::Ok.code());
        std::fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn term_query_multi_segment_happy_path_against_real_fixture() {
        let handle = open_reader();

        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(results_handle);
        assert!(!hits.is_empty());
        for pair in hits.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }

        ffi_close_scored_results(results_handle);
        ffi_close_directory_reader(handle);
    }

    #[test]
    fn boolean_query_multi_segment_happy_path_against_real_fixture() {
        let handle = open_reader();

        let should_field = "body";
        let should_term1 = b"cat";
        let should_term2 = b"bird";
        let should_fields = [
            should_field.as_ptr() as *const c_char,
            should_field.as_ptr() as *const c_char,
        ];
        let should_field_lens = [should_field.len(), should_field.len()];
        let should_terms = [should_term1.as_ptr(), should_term2.as_ptr()];
        let should_term_lens = [should_term1.len(), should_term2.len()];

        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_multi_segment(
                handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                should_fields.as_ptr(),
                should_field_lens.as_ptr(),
                should_terms.as_ptr(),
                should_term_lens.as_ptr(),
                2,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(results_handle);
        assert!(!hits.is_empty());
        for pair in hits.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }

        ffi_close_scored_results(results_handle);
        ffi_close_directory_reader(handle);
    }

    /// A `ScoredResultsHandle` passed where a `DirectoryReader` handle is
    /// expected must be rejected by the registry-tag check, not
    /// accidentally accepted -- see `handle.rs`'s `RegistryTag`.
    #[test]
    fn scored_results_handle_passed_as_reader_handle_is_invalid_handle() {
        let reader_handle = open_reader();
        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        assert_eq!(
            unsafe {
                ffi_search_term_query_multi_segment(
                    reader_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    10,
                    &mut results_handle as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );

        // Now use the *results* handle where a reader handle is expected.
        let mut second_results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                results_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut second_results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());

        ffi_close_scored_results(results_handle);
        ffi_close_directory_reader(reader_handle);
    }

    /// The reverse direction: a `DirectoryReader` handle passed where a
    /// `ScoredResultsHandle` is expected (`ffi_scored_results_len`) must
    /// also be rejected by the registry-tag check.
    #[test]
    fn directory_reader_handle_passed_as_scored_results_handle_is_invalid_handle() {
        let reader_handle = open_reader();
        let mut len: usize = 0;
        let rc = unsafe { ffi_scored_results_len(reader_handle, &mut len as *mut _) };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        ffi_close_directory_reader(reader_handle);
    }

    #[test]
    fn close_unknown_directory_reader_handle_is_invalid_handle() {
        assert_eq!(
            ffi_close_directory_reader(0x1234),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn double_close_directory_reader_is_invalid_handle_not_a_crash() {
        let handle = open_reader();
        assert_eq!(ffi_close_directory_reader(handle), FfiStatus::Ok.code());
        assert_eq!(
            ffi_close_directory_reader(handle),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn term_query_multi_segment_unknown_reader_handle_is_invalid_handle() {
        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn boolean_query_multi_segment_unknown_reader_handle_is_invalid_handle() {
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_multi_segment(
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
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn term_query_multi_segment_null_out_handle_is_null_pointer_error() {
        let handle = open_reader();
        let field = "body";
        let term = b"cat";
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_directory_reader(handle);
    }

    #[test]
    fn open_directory_reader_null_out_handle_is_null_pointer_error() {
        let path = fixture_dir_path();
        let rc = unsafe {
            ffi_open_directory_reader(
                path.as_ptr() as *const c_char,
                path.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
    }

    #[test]
    fn term_query_multi_segment_concurrent_happy_path_against_real_fixture() {
        let handle = open_reader();

        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment_concurrent(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(results_handle);
        assert!(!hits.is_empty());
        for pair in hits.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }

        ffi_close_scored_results(results_handle);
        ffi_close_directory_reader(handle);
    }

    #[test]
    fn boolean_query_multi_segment_concurrent_happy_path_against_real_fixture() {
        let handle = open_reader();

        let should_field = "body";
        let should_term1 = b"cat";
        let should_term2 = b"bird";
        let should_fields = [
            should_field.as_ptr() as *const c_char,
            should_field.as_ptr() as *const c_char,
        ];
        let should_field_lens = [should_field.len(), should_field.len()];
        let should_terms = [should_term1.as_ptr(), should_term2.as_ptr()];
        let should_term_lens = [should_term1.len(), should_term2.len()];

        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_multi_segment_concurrent(
                handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                should_fields.as_ptr(),
                should_field_lens.as_ptr(),
                should_terms.as_ptr(),
                should_term_lens.as_ptr(),
                2,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(results_handle);
        assert!(!hits.is_empty());
        for pair in hits.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }

        ffi_close_scored_results(results_handle);
        ffi_close_directory_reader(handle);
    }

    /// The critical proof for this task: the concurrent FFI wrapper must
    /// produce byte-for-byte identical `(doc_id, score)` output to the
    /// sequential FFI wrapper for the exact same input, over the FFI boundary
    /// itself (not just the underlying `lucene-search` functions, which
    /// `multi_segment.rs`'s own tests already cover) -- same real two-segment
    /// fixture both `term_query_multi_segment_happy_path_against_real_fixture`
    /// and `term_query_multi_segment_concurrent_happy_path_against_real_fixture`
    /// exercise individually, now compared directly against each other.
    #[test]
    fn term_query_multi_segment_concurrent_matches_sequential_ffi_wrapper() {
        let seq_handle = open_reader();
        let con_handle = open_reader();

        let field = "body";
        let term = b"cat";

        let mut seq_results: u64 = 0;
        assert_eq!(
            unsafe {
                ffi_search_term_query_multi_segment(
                    seq_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    10,
                    &mut seq_results as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );

        let mut con_results: u64 = 0;
        assert_eq!(
            unsafe {
                ffi_search_term_query_multi_segment_concurrent(
                    con_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    10,
                    &mut con_results as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );

        let seq_hits = read_scored_results(seq_results);
        let con_hits = read_scored_results(con_results);
        assert!(!seq_hits.is_empty());
        assert_eq!(
            seq_hits, con_hits,
            "concurrent FFI wrapper must match sequential FFI wrapper byte-for-byte"
        );

        ffi_close_scored_results(seq_results);
        ffi_close_scored_results(con_results);
        ffi_close_directory_reader(seq_handle);
        ffi_close_directory_reader(con_handle);
    }

    /// Same identical-output proof for the boolean-query concurrent wrapper.
    #[test]
    fn boolean_query_multi_segment_concurrent_matches_sequential_ffi_wrapper() {
        let seq_handle = open_reader();
        let con_handle = open_reader();

        let should_field = "body";
        let should_term1 = b"cat";
        let should_term2 = b"bird";
        let should_fields = [
            should_field.as_ptr() as *const c_char,
            should_field.as_ptr() as *const c_char,
        ];
        let should_field_lens = [should_field.len(), should_field.len()];
        let should_terms = [should_term1.as_ptr(), should_term2.as_ptr()];
        let should_term_lens = [should_term1.len(), should_term2.len()];

        let mut seq_results: u64 = 0;
        assert_eq!(
            unsafe {
                ffi_search_boolean_query_multi_segment(
                    seq_handle,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    should_fields.as_ptr(),
                    should_field_lens.as_ptr(),
                    should_terms.as_ptr(),
                    should_term_lens.as_ptr(),
                    2,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    10,
                    &mut seq_results as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );

        let mut con_results: u64 = 0;
        assert_eq!(
            unsafe {
                ffi_search_boolean_query_multi_segment_concurrent(
                    con_handle,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    should_fields.as_ptr(),
                    should_field_lens.as_ptr(),
                    should_terms.as_ptr(),
                    should_term_lens.as_ptr(),
                    2,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    10,
                    &mut con_results as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );

        let seq_hits = read_scored_results(seq_results);
        let con_hits = read_scored_results(con_results);
        assert!(!seq_hits.is_empty());
        assert_eq!(
            seq_hits, con_hits,
            "concurrent FFI wrapper must match sequential FFI wrapper byte-for-byte"
        );

        ffi_close_scored_results(seq_results);
        ffi_close_scored_results(con_results);
        ffi_close_directory_reader(seq_handle);
        ffi_close_directory_reader(con_handle);
    }

    #[test]
    fn term_query_multi_segment_concurrent_unknown_reader_handle_is_invalid_handle() {
        let field = "body";
        let term = b"cat";
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment_concurrent(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn boolean_query_multi_segment_concurrent_unknown_reader_handle_is_invalid_handle() {
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_multi_segment_concurrent(
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
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn term_query_multi_segment_concurrent_null_out_handle_is_null_pointer_error() {
        let handle = open_reader();
        let field = "body";
        let term = b"cat";
        let rc = unsafe {
            ffi_search_term_query_multi_segment_concurrent(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_directory_reader(handle);
    }

    #[test]
    fn boolean_query_multi_segment_concurrent_null_out_handle_is_null_pointer_error() {
        let handle = open_reader();
        let should_field = "body";
        let should_term = b"cat";
        let should_fields = [should_field.as_ptr() as *const c_char];
        let should_field_lens = [should_field.len()];
        let should_terms = [should_term.as_ptr()];
        let should_term_lens = [should_term.len()];
        let rc = unsafe {
            ffi_search_boolean_query_multi_segment_concurrent(
                handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                should_fields.as_ptr(),
                should_field_lens.as_ptr(),
                should_terms.as_ptr(),
                should_term_lens.as_ptr(),
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_directory_reader(handle);
    }

    /// Regression test for the exact bug class task #29's history flagged:
    /// a panic while a registry `Mutex` is held must poison-then-recover on
    /// the *next* call, not permanently wedge every future call into
    /// `FfiStatus::Panic`. Uses the thread-local
    /// `PANIC_ON_NEXT_MULTI_SEGMENT_QUERY` switch (armed and fired on this
    /// same test thread only, per this module's doc comment) rather than a
    /// process-wide flag, so a concurrently-running unrelated test can never
    /// observe or trigger it.
    #[test]
    fn registry_mutex_recovers_from_poisoning_after_a_panic_mid_multi_segment_query() {
        let handle = open_reader();
        let field = "body";
        let term = b"cat";

        arm_panic_on_next_multi_segment_query();
        let mut unused_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut unused_handle as *mut _, // valid, but never written: panics first.
            )
        };
        assert_eq!(rc, FfiStatus::Panic.code());

        // The registry mutex must have recovered: a normal call right after
        // must succeed, not report `FfiStatus::Panic` forever.
        let mut results_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_multi_segment(
                handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut results_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(!read_scored_results(results_handle).is_empty());

        ffi_close_scored_results(results_handle);
        ffi_close_directory_reader(handle);
    }
}
