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
//!
//! ## Scored variants (task #30) -- `ffi_search_term_query_scored`/
//! `ffi_search_boolean_query_scored`/`ffi_search_phrase_query_scored`
//!
//! ## MAXSCORE-pruned variants -- `ffi_search_term_query_scored_maxscore`/
//! `ffi_search_boolean_query_scored_maxscore`
//!
//! `ffi_search_term_query_scored_maxscore` is backed by real, block-level
//! MAXSCORE dynamic pruning (`lucene_search::search_term_query_scored_maxscore`,
//! which streams a single `TermQuery`'s postings through a `LazyDocsCursor`
//! and skips whole level-0 blocks a `TopDocsCollector`'s current worst kept
//! score proves are unreachable, instead of eagerly decoding every block). It
//! is scoped as narrowly as its Rust-level counterpart: single `TermQuery`
//! only.
//!
//! `ffi_search_boolean_query_scored_maxscore` extends this one level up:
//! `lucene_search::search_boolean_query_scored_maxscore` prunes a pure
//! SHOULD-disjunction `BooleanQuery` of plain `Clause::Term` clauses (no
//! `must`/`must_not`, `minimum_should_match <= 1`, every clause's term with
//! `docFreq > 1`) using a simplified two-tier essential/non-essential-style
//! MAXSCORE skip -- see that function's doc comment for the exact algorithm
//! and its honestly-documented scope narrowing versus a full multi-way WAND
//! pivot. Any `BooleanQuery` outside that scope (a `must`/`must_not` clause,
//! `minimum_should_match > 1`, a nested/non-term clause, or a singleton
//! `docFreq == 1` term) transparently falls back to the same exhaustive
//! `search_boolean_query_scored` path `ffi_search_boolean_query_scored`
//! itself calls, so the fast path never changes a result, only whether it's
//! reached faster.
//!
//! Every other scored function here has no competitive-score threshold at
//! all and never prunes.
//!
//! Same matching semantics as their unscored siblings above, but each feeds
//! the matched, live docs' real BM25 score (`lucene_search::similarity`) to a
//! [`lucene_search::TopDocsCollector`] (keeping only the best `top_n` hits,
//! see that type's doc comment for tie-breaking) instead of collecting every
//! match into a flat `Vec<i32>`. The resulting `(doc_id, score)` hits are
//! collected into a new [`crate::registry::ScoredResultsHandle`] -- a
//! separate registry/handle type from the unscored path's `ResultsHandle`,
//! see that struct's doc comment in `registry.rs` for why.
//!
//! **Norms**: [`open_field_norms`] looks a field's [`lucene_search::FieldNorms`]
//! up from the segment handle's `norms`/`norms_data` (populated by
//! [`crate::segment::ffi_open_segment`]'s optional `nvm_name`/`nvd_name`
//! parameters, also task #30) via `field_infos` name->number lookup, falling
//! back to `None` (real Lucene's `UNNORMED_FIELD_LENGTH` approximation) when
//! the segment was opened without norms, or the field itself has none --
//! exactly the same fallback `lucene_search`'s own scored functions already
//! document for a bare `norms: None`.
//!
//! **`ffi_search_boolean_query_scored`'s clause list** is the same flat,
//! `Clause::Term`-only four-parallel-array wire format as the unscored
//! `ffi_search_boolean_query` above (see that section's doc comment for why
//! nested/phrase clause construction isn't exposed over this C ABI yet) --
//! its norms map is built from every distinct field name appearing in
//! `must`/`should`/`must_not`'s flat term clauses.

use std::collections::HashMap;
use std::os::raw::c_char;

use lucene_codecs::postings::{DocInput, PosInput};
use lucene_search::field_norms::FieldNorms;
use lucene_search::{
    search_boolean_query, search_boolean_query_scored, search_boolean_query_scored_maxscore,
    search_phrase_query, search_phrase_query_scored, search_term_query, search_term_query_scored,
    search_term_query_scored_maxscore,
};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery, TopDocsCollector, VecCollector};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{
    lock_recovering, results, scored_results, segments, ResultsHandle, ScoredResultsHandle,
    SegmentHandle,
};

pub(crate) fn map_search_error(e: lucene_search::Error) -> FfiStatus {
    set_last_error(format!("search failed: {e}"));
    FfiStatus::Search
}

// Test-only panic-injection switch for
// `registry_mutex_recovers_from_poisoning_after_a_panic_mid_query` below:
// there is no way to reach a real internal `unwrap()`/indexing panic from
// adversarial-but-otherwise-well-formed bytes through this crate's public
// decode paths (every decoder here already turns corrupted bytes into an
// `Err` -> `FfiStatus::Decode`, per `segment.rs`'s garbage-bytes tests), so
// this flag fabricates the one thing a real panic there would have in
// common with any other panic: it fires *while `ffi_search_term_query`
// still holds the `segments()` registry's `MutexGuard`*, the exact
// condition that poisons the mutex. Never armed outside a test, and always
// disarmed (via `.replace(false)`) the instant it fires.
//
// **`thread_local!`, not a process-wide `static`** -- same reasoning as
// `PANIC_ON_NEXT_SCORED_TERM_QUERY` below: `cargo test` runs this crate's
// tests in parallel by default, and `ffi_search_term_query` is called by
// more than one test (this one arms it, but e.g.
// `scored_results_handle_rejected_by_unscored_results_accessors` also calls
// the unscored `ffi_search_term_query` for its own, unrelated assertions).
// A process-wide flag armed by this test could fire inside that other
// test's call if the two happened to run concurrently on separate threads,
// panicking a test that never armed anything -- exactly the intermittent
// failure this flag used to be exposed to. Scoping it `thread_local!`
// instead means arming and firing both happen on this test's own thread, so
// no other test's thread can ever observe or trigger it, regardless of
// scheduling.
#[cfg(test)]
thread_local! {
    static PANIC_ON_NEXT_TERM_QUERY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn arm_panic_on_next_term_query() {
    PANIC_ON_NEXT_TERM_QUERY.with(|c| c.set(true));
}

// Test-only panic-injection switch for
// `registry_mutex_recovers_from_poisoning_after_a_panic_mid_scored_query`
// below -- same purpose and same `thread_local!` shape as
// `PANIC_ON_NEXT_TERM_QUERY` above (see its doc comment for the race a
// process-wide `static` flag would otherwise expose this to: `cargo test`
// runs a crate's tests in parallel on a thread pool by default, and more
// than one test calls the function a shared flag would gate). Kept as its
// own flag rather than reusing `PANIC_ON_NEXT_TERM_QUERY` because it gates a
// different function (`ffi_search_term_query_scored`, not
// `ffi_search_term_query`).
#[cfg(test)]
thread_local! {
    static PANIC_ON_NEXT_SCORED_TERM_QUERY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn arm_panic_on_next_scored_term_query() {
    PANIC_ON_NEXT_SCORED_TERM_QUERY.with(|c| c.set(true));
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
        if PANIC_ON_NEXT_TERM_QUERY.with(|c| c.replace(false)) {
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
///
/// `pub(crate)` (rather than private) so [`crate::directory_reader`]'s
/// multi-segment boolean-query entry point (task #51) can reuse the exact
/// same flat-array clause decoder instead of duplicating it.
pub(crate) unsafe fn read_term_clauses(
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
            None,
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

/// Opens `field`'s [`FieldNorms`] against `segment`'s norms data, or `None`
/// when the segment was opened without a `.nvm`/`.nvd` pair
/// ([`crate::segment::ffi_open_segment`]'s `nvm_name`/`nvd_name`), or when
/// `field` itself has no norms entry (e.g. norms disabled for that field) --
/// both cases are the documented "fall back to
/// `lucene_search::similarity::UNNORMED_FIELD_LENGTH`" behavior, same as
/// passing `norms: None` directly to a `search_*_query_scored` function, not
/// an error.
///
/// **Recomputed on every call, not cached on `SegmentHandle`**:
/// `FieldNorms::open` scans every live doc in the field once to compute
/// `avgFieldLength` (see `field_norms.rs`'s doc comment) -- cheap relative to
/// a real query's own per-doc scoring work, and caching it would mean
/// threading a live-docs-aware invalidation story into `SegmentHandle` for a
/// shortcut this task's scope doesn't need; a future perf pass can revisit if
/// this ever shows up as measurably hot.
pub(crate) fn open_field_norms<'seg>(
    segment: &'seg SegmentHandle,
    field: &str,
) -> Result<Option<FieldNorms<'seg>>, FfiStatus> {
    let (Some(norms), Some(data)) = (segment.norms.as_ref(), segment.norms_data.as_deref()) else {
        return Ok(None);
    };
    let Some(field_info) = segment.field_infos.fields.iter().find(|f| f.name == field) else {
        return Ok(None);
    };
    let Some(entry) = norms.entry(field_info.number) else {
        return Ok(None);
    };
    let opened = FieldNorms::open(data, *entry, segment.max_doc, None).map_err(|e| {
        set_last_error(format!("opening norms for field {field}: {e}"));
        FfiStatus::Decode
    })?;
    Ok(Some(opened))
}

/// Scored sibling of [`ffi_search_term_query`]: runs `search_term_query_scored`
/// for `(field, term)`, keeping the best `top_n` `(doc_id, score)` hits (see
/// [`lucene_search::TopDocsCollector`]) in a new
/// [`crate::registry::ScoredResultsHandle`] written to
/// `*out_scored_results_handle` on success.
///
/// # Safety
/// `field` must be valid for `field_len` bytes, `term` for `term_len` bytes,
/// `out_scored_results_handle` valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_term_query_scored(
    segment_handle: u64,
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

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_term_query_scored: unknown or already-closed segment handle",
            );
            FfiStatus::InvalidHandle
        })?;

        // Test-only: see `arm_panic_on_next_scored_term_query`'s doc comment.
        // Fires while `segments` (the `MutexGuard` above) is still held, exactly
        // like a real decode panic reached through `search_term_query_scored`
        // below would.
        #[cfg(test)]
        if PANIC_ON_NEXT_SCORED_TERM_QUERY.with(|c| c.replace(false)) {
            panic!(
                "test-only simulated panic while the segments registry lock is held (scored path)"
            );
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
        let norms = open_field_norms(segment, &query.field)?;

        let mut collector = TopDocsCollector::new(top_n);
        search_term_query_scored(
            &segment.fields,
            doc_in.as_ref(),
            None,
            &query,
            norms.as_ref(),
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        });
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// MAXSCORE-pruned sibling of [`ffi_search_term_query_scored`]: same
/// `(field, term)`/`top_n`/[`ScoredResultsHandle`] contract, but runs
/// [`lucene_search::search_term_query_scored_maxscore`] instead of
/// [`search_term_query_scored`] -- streaming the term's postings through
/// [`lucene_codecs::postings::LazyDocsCursor`] and skipping whole level-0
/// blocks whose [`lucene_search::similarity::max_score_for_impacts`] upper
/// bound can't beat the [`TopDocsCollector`]'s current worst kept score
/// (once it's holding a full top-`n`), rather than eagerly decoding every
/// block via `DocInput::read_postings` the way `ffi_search_term_query_scored`
/// does. Produces byte-for-byte identical `top_docs()` results to
/// `ffi_search_term_query_scored` for the same query -- see that function's
/// Rust-level counterpart doc comment
/// ([`lucene_search::search_term_query_scored_maxscore`]) for the full
/// safety argument and its fallback cases (no `.doc` opened, `docFreq <= 1`,
/// or an index option `LazyDocsCursor` doesn't support all transparently
/// fall back to the exact same eager path `ffi_search_term_query_scored`
/// uses, never a silently different result).
///
/// `ffi_search_boolean_query_scored_maxscore` (further below) is this
/// module's other MAXSCORE-pruned entry point, one level up (a pure
/// SHOULD-disjunction `BooleanQuery` of plain term clauses, see that
/// function's doc comment for its own scope). `ffi_search_boolean_query_scored`
/// itself (below) still sums per-clause BM25 scores over an eagerly-resolved
/// matched-doc set with no competitive-score threshold at all -- unchanged --
/// and none of this module's other scored functions consult
/// `min_competitive_score()` either.
///
/// # Safety
/// Same contract as [`ffi_search_term_query_scored`]'s.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_term_query_scored_maxscore(
    segment_handle: u64,
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

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_term_query_scored_maxscore: unknown or already-closed segment handle",
            );
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
        let norms = open_field_norms(segment, &query.field)?;

        let mut collector = TopDocsCollector::new(top_n);
        search_term_query_scored_maxscore(
            &segment.fields,
            doc_in.as_ref(),
            None,
            &query,
            norms.as_ref(),
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        });
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// Scored sibling of [`ffi_search_boolean_query`]: same flat, `Clause::Term`-only
/// four-parallel-array clause wire format (see this module's doc comment), but
/// keeps the best `top_n` `(doc_id, score)` hits (each matched doc's score is the
/// sum of its BM25 score across every satisfied `must`/`should` clause, see
/// [`lucene_search::search_boolean_query_scored`]'s doc comment) in a new
/// [`crate::registry::ScoredResultsHandle`].
///
/// # Safety
/// Same contract as [`ffi_search_boolean_query`]'s, plus `out_scored_results_handle`
/// must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_boolean_query_scored(
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

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_boolean_query_scored: unknown or already-closed segment handle",
            );
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

        // Norms map keyed by every distinct field name across `must`/`should`
        // (see this module's doc comment) -- a field with no norms entry (or no
        // opened norms at all) is simply absent from the map, which
        // `search_boolean_query_scored` treats as "fall back to
        // `UNNORMED_FIELD_LENGTH` for that field", same as `open_field_norms`'s
        // own `None` fallback.
        let mut field_names: Vec<&str> = query
            .must
            .iter()
            .chain(query.should.iter())
            .chain(query.must_not.iter())
            .filter_map(|c| match c {
                Clause::Term(t) => Some(t.field.as_str()),
                Clause::Phrase(_)
                | Clause::Boolean(_)
                | Clause::DisjunctionMax(_)
                | Clause::ConstantScore(_)
                | Clause::Boost(_)
                | Clause::Wildcard(_)
                | Clause::Prefix(_)
                | Clause::Fuzzy(_)
                | Clause::Regexp(_)
                | Clause::Span(_)
                | Clause::PointsRange(_)
                | Clause::MatchAllDocs(_)
                | Clause::MatchNoDocs(_)
                | Clause::TermInSet(_) => None,
            })
            .collect();
        field_names.sort_unstable();
        field_names.dedup();
        let mut norms_map: HashMap<String, FieldNorms<'_>> = HashMap::new();
        for name in field_names {
            if let Some(field_norms) = open_field_norms(segment, name)? {
                norms_map.insert(name.to_string(), field_norms);
            }
        }
        let norms_arg = (!norms_map.is_empty()).then_some(&norms_map);

        let mut collector = TopDocsCollector::new(top_n);
        search_boolean_query_scored(
            &segment.fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &query,
            norms_arg,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        });
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// MAXSCORE-pruned sibling of [`ffi_search_boolean_query_scored`]: same flat,
/// four-parallel-array clause wire format and the exact same
/// [`ScoredResultsHandle`] contract, but runs
/// [`lucene_search::search_boolean_query_scored_maxscore`] instead of
/// [`search_boolean_query_scored`] -- streaming every `should` clause's
/// postings through its own [`lucene_codecs::postings::LazyDocsCursor`] and
/// skipping whole level-0 blocks that a per-clause bound proves can never
/// beat the [`TopDocsCollector`]'s current worst kept score, rather than
/// eagerly materializing the whole matched-doc set the way
/// `ffi_search_boolean_query_scored` does. See
/// [`lucene_search::search_boolean_query_scored_maxscore`]'s own doc comment
/// for:
/// - the exact fast-path preconditions (pure SHOULD disjunction, no nested
///   clauses, every clause a plain `Clause::Term` with `docFreq > 1`,
///   `minimum_should_match <= 1`) -- any query outside that scope
///   transparently falls back to calling `search_boolean_query_scored`
///   verbatim, the same function `ffi_search_boolean_query_scored` calls, so
///   this entry point never produces a different result for a query the
///   fast path can't handle;
/// - the honestly-scoped simplification this function's pruning actually
///   implements (a two-tier essential/non-essential-style skip driven by a
///   real, always-valid per-clause global score bound, not a full
///   multi-way WAND pivot).
///
/// Produces byte-for-byte identical `top_docs()` results to
/// [`ffi_search_boolean_query_scored`] for the same query, for every
/// `must`/`should`/`must_not` combination -- including ones outside the fast
/// path's scope, which simply take the identical eager code path under the
/// hood.
///
/// # Safety
/// Same contract as [`ffi_search_boolean_query_scored`]'s.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_boolean_query_scored_maxscore(
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

        let segments = lock_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_boolean_query_scored_maxscore: unknown or already-closed segment \
                 handle",
            );
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

        // Same norms-map construction as `ffi_search_boolean_query_scored` above
        // -- see that function's comment for why a field with no norms entry is
        // simply absent from the map.
        let mut field_names: Vec<&str> = query
            .must
            .iter()
            .chain(query.should.iter())
            .chain(query.must_not.iter())
            .filter_map(|c| match c {
                Clause::Term(t) => Some(t.field.as_str()),
                Clause::Phrase(_)
                | Clause::Boolean(_)
                | Clause::DisjunctionMax(_)
                | Clause::ConstantScore(_)
                | Clause::Boost(_)
                | Clause::Wildcard(_)
                | Clause::Prefix(_)
                | Clause::Fuzzy(_)
                | Clause::Regexp(_)
                | Clause::Span(_)
                | Clause::PointsRange(_)
                | Clause::MatchAllDocs(_)
                | Clause::MatchNoDocs(_)
                | Clause::TermInSet(_) => None,
            })
            .collect();
        field_names.sort_unstable();
        field_names.dedup();
        let mut norms_map: HashMap<String, FieldNorms<'_>> = HashMap::new();
        for name in field_names {
            if let Some(field_norms) = open_field_norms(segment, name)? {
                norms_map.insert(name.to_string(), field_norms);
            }
        }
        let norms_arg = (!norms_map.is_empty()).then_some(&norms_map);

        let mut collector = TopDocsCollector::new(top_n);
        search_boolean_query_scored_maxscore(
            &segment.fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &query,
            norms_arg,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        });
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// Scored sibling of [`ffi_search_phrase_query`]: same single-field, in-phrase-order
/// term list wire format, but keeps the best `top_n` `(doc_id, score)` hits (via
/// [`lucene_search::search_phrase_query_scored`]) in a new
/// [`crate::registry::ScoredResultsHandle`]. Same `.pos`-file requirement for a
/// multi-term phrase as the unscored sibling -- see that function's doc comment.
///
/// # Safety
/// Same contract as [`ffi_search_phrase_query`]'s, plus `out_scored_results_handle`
/// must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_phrase_query_scored(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    terms: *const *const u8,
    term_lens: *const usize,
    term_count: usize,
    top_n: usize,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
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
            set_last_error(
                "ffi_search_phrase_query_scored: unknown or already-closed segment handle",
            );
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
        let norms = open_field_norms(segment, &query.field)?;

        let mut collector = TopDocsCollector::new(top_n);
        search_phrase_query_scored(
            &segment.fields,
            doc_in.as_ref(),
            pos_in.as_ref(),
            None,
            None,
            &query,
            norms.as_ref(),
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = lock_recovering(scored_results()).insert(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        });
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
        let hex = "bea914ffd84e035aaac43aca30240b47";
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
        open_segment_with_norms(dir_handle, with_pos, false)
    }

    /// Same as [`open_segment`], optionally also opening this fixture's real
    /// `_0.nvm`/`_0.nvd` (task #30) so scored-query tests can exercise the
    /// real-norms path.
    fn open_segment_with_norms(dir_handle: u64, with_pos: bool, with_norms: bool) -> u64 {
        let fnm = "_0.fnm";
        let tim = "_0_Lucene104_0.tim";
        let tip = "_0_Lucene104_0.tip";
        let tmd = "_0_Lucene104_0.tmd";
        let doc = "_0_Lucene104_0.doc";
        let pos = "_0_Lucene104_0.pos";
        let nvm = "_0.nvm";
        let nvd = "_0.nvd";
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
                if with_norms {
                    nvm.as_ptr() as *const c_char
                } else {
                    std::ptr::null()
                },
                if with_norms { nvm.len() } else { 0 },
                if with_norms {
                    nvd.as_ptr() as *const c_char
                } else {
                    std::ptr::null()
                },
                if with_norms { nvd.len() } else { 0 },
                std::ptr::null(), // dvm_name: not needed by any scored-query test
                0,
                std::ptr::null(), // dvd_name
                0,
                std::ptr::null(), // dv_suffix
                0,
                std::ptr::null(), // kdm_name: no points data needed by this test/call
                0,
                std::ptr::null(), // kdi_name
                0,
                std::ptr::null(), // kdd_name
                0,
                id.as_ptr(),
                suffix.as_ptr() as *const c_char,
                suffix.len(),
                8959,
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

    // ---- Scored query tests (task #30) ----

    use crate::results_scored::{ffi_close_scored_results, ffi_scored_results_copy};

    fn read_scored_results(scored_results_handle: u64) -> Vec<(i32, f32)> {
        let mut len: usize = 0;
        assert_eq!(
            unsafe {
                crate::results_scored::ffi_scored_results_len(
                    scored_results_handle,
                    &mut len as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );
        let mut doc_ids = vec![0i32; len];
        let mut scores = vec![0.0f32; len];
        assert_eq!(
            unsafe {
                ffi_scored_results_copy(
                    scored_results_handle,
                    doc_ids.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    len,
                )
            },
            FfiStatus::Ok.code()
        );
        doc_ids.into_iter().zip(scores).collect()
    }

    /// Reimplements the expected unnormed (`norms: None`-fallback) BM25 score
    /// independently of `similarity::score` -- same "recompute the expected
    /// value, don't just call the function under test and trust it" approach
    /// `lucene-search`'s own `scoring_fixtures.rs` uses -- from this fixture's
    /// known real postings stats (`manifest.properties`: `body`'s `docFreq`/
    /// `docCount`/per-doc `freq`, e.g. `cat`'s `docFreq=2`, `body.docCount=4`,
    /// `postingsDocs=0,2`/`postingsFreqs=2,1`).
    fn expected_unnormed_bm25(doc_freq: i64, doc_count: i64, freq: f32) -> f32 {
        lucene_search::similarity::idf(doc_freq, doc_count)
            * lucene_search::similarity::tf_norm(
                freq,
                lucene_search::similarity::UNNORMED_FIELD_LENGTH,
                lucene_search::similarity::UNNORMED_FIELD_LENGTH,
                lucene_search::similarity::DEFAULT_K1,
                lucene_search::similarity::DEFAULT_B,
            )
    }

    #[test]
    fn term_query_scored_body_cat_returns_expected_docs_and_scores() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let hits = read_scored_results(scored_handle);
        // cat: docFreq=2, body.docCount=4; doc 0 has freq 2, doc 2 has freq 1
        // (manifest.properties) -- with no norms opened, both fall back to
        // `UNNORMED_FIELD_LENGTH`, so only `freq` differs between the two hits.
        let expected_doc0 = expected_unnormed_bm25(2, 4, 2.0);
        let expected_doc2 = expected_unnormed_bm25(2, 4, 1.0);
        // Higher freq (doc 0) scores strictly higher, so best-first order puts
        // doc 0 ahead of doc 2.
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 0);
        assert!((hits[0].1 - expected_doc0).abs() < 1e-4);
        assert_eq!(hits[1].0, 2);
        assert!((hits[1].1 - expected_doc2).abs() < 1e-4);

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_scored_top_n_keeps_only_the_best_hit() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                1, // top_n
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(scored_handle);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0); // doc 0 (freq 2) outscores doc 2 (freq 1).

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_scored_missing_term_returns_empty_results_not_an_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"zzz-missing";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_scored_results(scored_handle).is_empty());

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Scoring a second, non-`body` field (`id`) against a segment opened with
    /// real norms must also succeed -- `open_field_norms`'s field lookup is
    /// keyed by field name/number per call, not hardcoded to `body`.
    #[test]
    fn term_query_scored_non_body_field_with_real_norms_succeeds() {
        let dir_handle = open_dir();
        let seg_handle = open_segment_with_norms(dir_handle, false, true);

        let field = "id";
        let term = b"id2";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(scored_handle);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// `open_field_norms`'s "field not present in `field_infos` at all" branch:
    /// a field name this segment's `.fnm` never declared must still be a
    /// well-formed (empty-results) scored query, not an error, whether or not
    /// the segment has norms opened.
    #[test]
    fn term_query_scored_field_not_in_field_infos_falls_back_to_unnormed() {
        let dir_handle = open_dir();
        let seg_handle = open_segment_with_norms(dir_handle, false, true);

        let field = "no-such-field";
        let term = b"whatever";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_scored_results(scored_handle).is_empty());

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Proves this crate's MAXSCORE-pruned entry point,
    /// `ffi_search_term_query_scored_maxscore`, both (a) returns results
    /// identical to the naive/eager FFI path (`ffi_search_term_query_scored`)
    /// and (b) genuinely exercises real level-0 block skipping *through the
    /// FFI boundary* -- not just that `lucene_search`'s own Rust-level unit
    /// test (`maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks`
    /// in `lucene-search/src/lib.rs`) already proves the underlying function
    /// prunes; that's a strictly weaker claim than "the FFI caller actually
    /// reaches the pruned path".
    ///
    /// Uses the same real, Java-written fixture term as that Rust-level test:
    /// `big`/`"everywhere"`, `docFreq == 300` (one full 256-doc level-0 block
    /// with real impacts, plus a 44-doc tail), opened here with real per-doc
    /// norms (same reasoning as that test's doc comment: the impacts' bound
    /// is only a valid upper bound against the score formula that consumes
    /// those same real norm bytes).
    ///
    /// The skip-happened signal is `lucene_search::test_only_maxscore_block_skip_counter`
    /// itself -- reused verbatim across the crate boundary rather than
    /// reimplemented, made reachable here only via this crate's
    /// `[dev-dependencies]` edge enabling `lucene-search`'s `test-support`
    /// feature (see both `Cargo.toml`s), which normal (non-test) builds of
    /// this crate never enable.
    #[test]
    fn term_query_scored_maxscore_matches_eager_ffi_path_and_actually_skips_blocks() {
        let dir_handle = open_dir();
        let seg_handle = open_segment_with_norms(dir_handle, false, true);

        let field = "big";
        let term = b"everywhere";

        for &top_n in &[1usize, 5, 50, 300] {
            let mut eager_handle: u64 = 0;
            let rc = unsafe {
                ffi_search_term_query_scored(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    top_n,
                    &mut eager_handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            let eager_hits = read_scored_results(eager_handle);

            lucene_search::test_only_maxscore_block_skip_counter::reset();
            let mut maxscore_handle: u64 = 0;
            let rc = unsafe {
                ffi_search_term_query_scored_maxscore(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    top_n,
                    &mut maxscore_handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            let maxscore_hits = read_scored_results(maxscore_handle);
            let skips = lucene_search::test_only_maxscore_block_skip_counter::count();

            assert_eq!(
                eager_hits, maxscore_hits,
                "top_{top_n}: MAXSCORE FFI path must match the eager FFI path exactly"
            );

            if top_n < 300 {
                assert!(
                    skips > 0,
                    "top_{top_n}: should reach the block's best-scoring combination \
                     within its first few docs, making the rest of the block safely \
                     skippable through the FFI boundary (got {skips} skips)"
                );
            } else {
                assert_eq!(
                    skips, 0,
                    "top_{top_n} == the full docFreq: nothing should be skippable \
                     (got {skips} skips)"
                );
            }

            ffi_close_scored_results(eager_handle);
            ffi_close_scored_results(maxscore_handle);
        }

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// `ffi_search_term_query_scored_maxscore`'s unknown-segment-handle branch
    /// must behave exactly like `ffi_search_term_query_scored`'s: an
    /// `InvalidHandle` error, not a panic/UB, and `*out_scored_results_handle`
    /// left untouched.
    #[test]
    fn term_query_scored_maxscore_unknown_segment_handle_is_an_error() {
        let field = "big";
        let term = b"everywhere";
        let mut scored_handle: u64 = 0xDEAD_u64;
        let rc = unsafe {
            ffi_search_term_query_scored_maxscore(
                999_999, // never allocated
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        assert_eq!(scored_handle, 0xDEAD_u64);
    }

    /// `ffi_search_term_query_scored_maxscore`'s missing-term branch must
    /// return an empty, well-formed results handle (falls back to
    /// `search_term_query_scored_maxscore`'s own early return for an unknown
    /// field/term), not an error.
    #[test]
    fn term_query_scored_maxscore_missing_term_returns_empty_results_not_an_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"zzz-missing";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored_maxscore(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_scored_results(scored_handle).is_empty());

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Proves this crate's `ffi_search_boolean_query_scored_maxscore` entry
    /// point, analogous to
    /// `term_query_scored_maxscore_matches_eager_ffi_path_and_actually_skips_blocks`
    /// one level up: a three-clause pure-SHOULD `BooleanQuery`
    /// (`big`/`"everywhere"`, `body`/`"cat"`, `body`/`"dog"` -- the same
    /// three-clause fixture `lucene_search`'s own
    /// `boolean_maxscore_lazy_path_matches_eager_path_on_real_fixture`/
    /// `test_only_boolean_maxscore_block_skip_counter_records_real_skips`
    /// unit tests already prove skip block decode for at the Rust level)
    /// both (a) returns results identical to
    /// `ffi_search_boolean_query_scored` and (b) genuinely reaches real
    /// level-0 block skipping *through the FFI boundary*, via the same
    /// `lucene_search::test_only_maxscore_block_skip_counter` reused across
    /// the crate boundary (see
    /// `term_query_scored_maxscore_matches_eager_ffi_path_and_actually_skips_blocks`'s
    /// doc comment for why that's a strictly stronger claim than the
    /// Rust-level unit test alone).
    #[test]
    fn boolean_query_scored_maxscore_matches_eager_ffi_path_and_actually_skips_blocks() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let big_field = "big";
        let big_term = b"everywhere";
        let cat_field = "body";
        let cat_term = b"cat";
        let dog_field = "body";
        let dog_term = b"dog";

        let should_fields = [
            big_field.as_ptr() as *const c_char,
            cat_field.as_ptr() as *const c_char,
            dog_field.as_ptr() as *const c_char,
        ];
        let should_field_lens = [big_field.len(), cat_field.len(), dog_field.len()];
        let should_terms = [big_term.as_ptr(), cat_term.as_ptr(), dog_term.as_ptr()];
        let should_term_lens = [big_term.len(), cat_term.len(), dog_term.len()];

        for &top_n in &[1usize, 2, 5, 20, 9000] {
            let mut eager_handle: u64 = 0;
            let rc = unsafe {
                ffi_search_boolean_query_scored(
                    seg_handle,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    should_fields.as_ptr(),
                    should_field_lens.as_ptr(),
                    should_terms.as_ptr(),
                    should_term_lens.as_ptr(),
                    3,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    top_n,
                    &mut eager_handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            let eager_hits = read_scored_results(eager_handle);

            lucene_search::test_only_maxscore_block_skip_counter::reset();
            let mut maxscore_handle: u64 = 0;
            let rc = unsafe {
                ffi_search_boolean_query_scored_maxscore(
                    seg_handle,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    should_fields.as_ptr(),
                    should_field_lens.as_ptr(),
                    should_terms.as_ptr(),
                    should_term_lens.as_ptr(),
                    3,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    top_n,
                    &mut maxscore_handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            let maxscore_hits = read_scored_results(maxscore_handle);

            assert_eq!(
                eager_hits, maxscore_hits,
                "top_{top_n}: boolean MAXSCORE FFI path must match the eager FFI path exactly"
            );

            ffi_close_scored_results(eager_handle);
            ffi_close_scored_results(maxscore_handle);
        }

        // A small top_n should have reached at least one real per-clause
        // block skip through the FFI boundary somewhere across the loop
        // above (the last iteration run was top_n == 9000, so re-run top_n
        // == 1 alone here to check the skip counter deterministically).
        lucene_search::test_only_maxscore_block_skip_counter::reset();
        let mut maxscore_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_scored_maxscore(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                should_fields.as_ptr(),
                should_field_lens.as_ptr(),
                should_terms.as_ptr(),
                should_term_lens.as_ptr(),
                3,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                1,
                &mut maxscore_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let skips = lucene_search::test_only_maxscore_block_skip_counter::count();
        assert!(
            skips > 0,
            "top_1 should make at least one clause's block provably \
             uncompetitive through the FFI boundary (got {skips} skips)"
        );
        ffi_close_scored_results(maxscore_handle);

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// `ffi_search_boolean_query_scored_maxscore`'s unknown-segment-handle
    /// branch must behave exactly like `ffi_search_boolean_query_scored`'s.
    #[test]
    fn boolean_query_scored_maxscore_unknown_segment_handle_is_invalid_handle() {
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_scored_maxscore(
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
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    /// `ffi_search_boolean_query_scored_maxscore`'s null-out-pointer branch
    /// must behave exactly like `ffi_search_boolean_query_scored`'s.
    #[test]
    fn boolean_query_scored_maxscore_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let rc = unsafe {
            ffi_search_boolean_query_scored_maxscore(
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
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Falls back to the eager path unchanged when a `must` clause is
    /// present -- `ffi_search_boolean_query_scored_maxscore`'s fast-path
    /// precondition (see its doc comment) excludes `must`/`must_not`
    /// entirely.
    #[test]
    fn boolean_query_scored_maxscore_falls_back_when_must_clause_present() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let must_field = "body";
        let must_term = b"cat";
        let must_fields = [must_field.as_ptr() as *const c_char];
        let must_field_lens = [must_field.len()];
        let must_terms = [must_term.as_ptr()];
        let must_term_lens = [must_term.len()];

        let should_field = "body";
        let should_term = b"dog";
        let should_fields = [should_field.as_ptr() as *const c_char];
        let should_field_lens = [should_field.len()];
        let should_terms = [should_term.as_ptr()];
        let should_term_lens = [should_term.len()];

        let mut eager_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_scored(
                seg_handle,
                must_fields.as_ptr(),
                must_field_lens.as_ptr(),
                must_terms.as_ptr(),
                must_term_lens.as_ptr(),
                1,
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
                &mut eager_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut maxscore_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_scored_maxscore(
                seg_handle,
                must_fields.as_ptr(),
                must_field_lens.as_ptr(),
                must_terms.as_ptr(),
                must_term_lens.as_ptr(),
                1,
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
                &mut maxscore_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        assert_eq!(
            read_scored_results(eager_handle),
            read_scored_results(maxscore_handle)
        );

        ffi_close_scored_results(eager_handle);
        ffi_close_scored_results(maxscore_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Differential proof that `ffi_open_segment`'s `nvm_name`/`nvd_name`
    /// parameters actually reach `search_term_query_scored`'s `norms` argument:
    /// real per-doc field lengths (doc 0 length 3, doc 2 length 1, avg 2.25 --
    /// same fixture values `crates/lucene-search/tests/scoring_fixtures.rs`
    /// independently verifies against real Lucene-written norm bytes) must
    /// yield different scores than the `None`-fallback constant-length path
    /// exercised by `term_query_scored_body_cat_returns_expected_docs_and_scores`
    /// above.
    #[test]
    fn term_query_scored_with_real_norms_differs_from_unnormed_fallback() {
        let dir_handle = open_dir();
        let seg_handle = open_segment_with_norms(dir_handle, false, true);

        let field = "body";
        let term = b"cat";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(scored_handle);
        assert_eq!(hits.len(), 2);

        let avg = 2.25f32; // (3 + 2 + 1 + 3) / 4, see this test's doc comment.
        let expected_doc0 = lucene_search::similarity::idf(2, 4)
            * lucene_search::similarity::tf_norm(
                2.0,
                3.0,
                avg,
                lucene_search::similarity::DEFAULT_K1,
                lucene_search::similarity::DEFAULT_B,
            );
        let expected_doc2 = lucene_search::similarity::idf(2, 4)
            * lucene_search::similarity::tf_norm(
                1.0,
                1.0,
                avg,
                lucene_search::similarity::DEFAULT_K1,
                lucene_search::similarity::DEFAULT_B,
            );
        let by_doc = |doc_id: i32| hits.iter().find(|h| h.0 == doc_id).unwrap().1;
        assert!((by_doc(0) - expected_doc0).abs() < 1e-4);
        assert!((by_doc(2) - expected_doc2).abs() < 1e-4);

        // And it must genuinely differ from the unnormed fallback -- otherwise
        // this test wouldn't be distinguishing "norms wired through" from
        // "norms silently ignored".
        let unnormed_doc0 = expected_unnormed_bm25(2, 4, 2.0);
        assert!((by_doc(0) - unnormed_doc0).abs() > 1e-4);

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_scored_unknown_segment_handle_is_invalid_handle() {
        let field = "body";
        let term = b"cat";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    /// A directory handle passed where a segment handle is expected must be
    /// rejected by the registry-tag check, same as the unscored sibling's
    /// equivalent test above.
    #[test]
    fn term_query_scored_directory_handle_passed_as_segment_handle_is_invalid_handle() {
        let dir_handle = open_dir();
        let field = "body";
        let term = b"cat";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                dir_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_scored_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let term = b"cat";
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_scored_must_cat_must_not_bird_matches_expected_doc() {
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

        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_scored(
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
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(scored_handle);
        // Same matched set as the unscored sibling test: body/cat -> [0, 2];
        // must_not (bird -> [1, 4]) removes neither. Score is `cat`'s own BM25
        // (the only scoring clause), same values as the plain term-scored test.
        assert_eq!(hits.len(), 2);
        let expected_doc0 = expected_unnormed_bm25(2, 4, 2.0);
        let expected_doc2 = expected_unnormed_bm25(2, 4, 1.0);
        let by_doc = |doc_id: i32| hits.iter().find(|h| h.0 == doc_id).unwrap().1;
        assert!((by_doc(0) - expected_doc0).abs() < 1e-4);
        assert!((by_doc(2) - expected_doc2).abs() < 1e-4);

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_scored_no_clauses_matches_nothing() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_scored(
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
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_scored_results(scored_handle).is_empty());
        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn boolean_query_scored_unknown_segment_handle_is_invalid_handle() {
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_boolean_query_scored(
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
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn boolean_query_scored_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let rc = unsafe {
            ffi_search_boolean_query_scored(
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
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_scored_single_term_delegates_to_term_query() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat".as_slice();
        let terms = [term.as_ptr()];
        let term_lens = [term.len()];

        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                terms.as_ptr(),
                term_lens.as_ptr(),
                1,
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(scored_handle);
        assert_eq!(hits.len(), 2);
        let expected_doc0 = expected_unnormed_bm25(2, 4, 2.0);
        let expected_doc2 = expected_unnormed_bm25(2, 4, 1.0);
        let by_doc = |doc_id: i32| hits.iter().find(|h| h.0 == doc_id).unwrap().1;
        assert!((by_doc(0) - expected_doc0).abs() < 1e-4);
        assert!((by_doc(2) - expected_doc2).abs() < 1e-4);

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_scored_multi_term_without_pos_file_is_search_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false); // no .pos opened

        let field = "body";
        let t1 = b"cat".as_slice();
        let t2 = b"dog".as_slice();
        let terms = [t1.as_ptr(), t2.as_ptr()];
        let term_lens = [t1.len(), t2.len()];

        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                terms.as_ptr(),
                term_lens.as_ptr(),
                2,
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Search.code());

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_scored_empty_terms_matches_nothing() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(read_scored_results(scored_handle).is_empty());
        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn phrase_query_scored_unknown_segment_handle_is_invalid_handle() {
        let field = "body";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_phrase_query_scored(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn phrase_query_scored_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let rc = unsafe {
            ffi_search_phrase_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Regression test for the mutex-poisoning fix, exercised for the scored
    /// query path (task #30): mirrors
    /// `registry_mutex_recovers_from_poisoning_after_a_panic_mid_query` above,
    /// but for `ffi_search_term_query_scored`, using its own thread-local
    /// panic-injection switch -- see `arm_panic_on_next_scored_term_query`'s
    /// doc comment for why both this and the unscored path's switch are
    /// thread-local (both were once a shared process-global `AtomicBool`,
    /// which raced with unrelated tests calling the same FFI entry point
    /// concurrently under `cargo test`'s default parallel execution).
    #[test]
    fn registry_mutex_recovers_from_poisoning_after_a_panic_mid_scored_query() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";

        arm_panic_on_next_scored_term_query();
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
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
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(scored_handle);
        assert_eq!(hits.len(), 2);

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// A scored-results handle must be rejected by the unscored
    /// `ffi_results_len`/`ffi_results_copy`/`ffi_close_results` path, and vice
    /// versa -- the two registries' `RegistryTag`s (`Results` vs
    /// `ScoredResults`) must keep them from aliasing each other, same as any
    /// other cross-registry handle-tag test in this crate.
    #[test]
    fn scored_results_handle_rejected_by_unscored_results_accessors() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let term = b"cat";

        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_results_len(scored_handle, &mut len as *mut _) },
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(
            ffi_close_results(scored_handle),
            FfiStatus::InvalidHandle.code()
        );

        // And the reverse: an unscored results handle rejected by the scored
        // accessors.
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
        assert_eq!(
            unsafe {
                crate::results_scored::ffi_scored_results_len(results_handle, &mut len as *mut _)
            },
            FfiStatus::InvalidHandle.code()
        );
        assert_eq!(
            ffi_close_scored_results(results_handle),
            FfiStatus::InvalidHandle.code()
        );

        ffi_close_scored_results(scored_handle);
        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }
}
