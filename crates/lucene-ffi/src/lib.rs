//! lucene-ffi: the C-ABI/JNI boundary for this port's query execution path.
//!
//! ## Scope of this task (see PLAN.md's Phase 4 section for the full plan)
//!
//! This is the first real FFI surface in this workspace, wrapping this
//! port's existing `search_term_query`/`search_boolean_query`/
//! `search_phrase_query` (`crates/lucene-search`) so a JVM caller (a
//! separate JNI wrapper class, not part of this Rust repo) can open a
//! filesystem directory, open one segment's already-decoded term
//! dictionary and postings files, run a query, and read the matching doc
//! IDs back out — all through opaque `u64` handles, `catch_unwind`-guarded
//! `extern "C" fn`s, and status codes, per the `ffi-safety` skill.
//!
//! **In scope:**
//! - [`directory::ffi_open_directory`]/[`directory::ffi_close_directory`]:
//!   opens/closes a real `lucene_store::directory::FsDirectory`.
//! - [`segment::ffi_open_segment`]/[`segment::ffi_close_segment`]: opens
//!   one segment's `.fnm`/`.tim`/`.tip`/`.tmd` term dictionary
//!   (`lucene_codecs::blocktree::open`) plus, optionally, its `.doc`/`.pos`
//!   postings files (needed for `docFreq > 1` terms and multi-term phrase
//!   queries respectively).
//! - [`query::ffi_search_term_query`]/[`query::ffi_search_boolean_query`]/
//!   [`query::ffi_search_phrase_query`]: run the matching
//!   `lucene_search::search_*_query` function, collecting every matching
//!   live doc ID (via a plain `lucene_search::VecCollector`, entirely
//!   Rust-side) into a new results handle.
//! - [`results::ffi_results_len`]/[`results::ffi_results_copy`]/
//!   [`results::ffi_close_results`]: reads a results handle's doc IDs back
//!   out via a caller-allocated buffer (bulk copy, not a per-index
//!   accessor — see `results.rs`'s module doc for why), then releases it.
//! - [`query::ffi_search_term_query_scored`]/[`query::ffi_search_boolean_query_scored`]/
//!   [`query::ffi_search_phrase_query_scored`] (task #30): scored siblings of the
//!   three query functions above, keeping the best `top_n` `(doc_id, score)` BM25
//!   hits (via `lucene_search::TopDocsCollector`) in a new
//!   [`registry::ScoredResultsHandle`] — see `query.rs`'s module doc for the norms
//!   plumbing and [`segment::ffi_open_segment`]'s `nvm_name`/`nvd_name` parameters
//!   for how a segment's real per-doc/avg field lengths reach these functions.
//! - [`results_scored::ffi_scored_results_len`]/[`results_scored::ffi_scored_results_copy`]/
//!   [`results_scored::ffi_close_scored_results`]: reads a scored results handle's
//!   `(doc_id, score)` hits back out via two caller-allocated parallel buffers (see
//!   `results_scored.rs`'s module doc for why parallel buffers, not one interleaved
//!   one), then releases it.
//! - [`error::guard`]/[`ffi_get_last_error_message`]: every exported
//!   function's panic-safety wrapper and the thread-local last-error
//!   message accessor.
//!
//! **Deliberately deferred, tracked in `docs/parity.md`:**
//! - **`.liv` (live docs / deletions) support** — every query call here
//!   passes `live_docs: None` to `lucene_search`'s functions (this port's
//!   fixture segment has no deletions, and `lucene_search`'s own contract
//!   already treats `None` as "no deletions" as its documented, correct
//!   behavior — not a shortcut on top of it). Wiring a `.liv` file open
//!   into `SegmentHandle` and threading `Option<&FixedBitSet>` through is a
//!   small, mechanical follow-up once needed.
//! - **`.pay` (payloads) for phrase queries** — `ffi_open_segment` has no
//!   `pay_name` parameter yet; `search_phrase_query` accepts `pay_in:
//!   Option<&PayInput>` and this crate always passes `None`, which is
//!   correct for a field with no payloads and a hard error surfaced as
//!   [`error::FfiStatus::Search`] for one that needs it.
//! - **Multi-segment search / a unified `.si`-driven "open everything"
//!   entry point** — `ffi_open_segment` takes already-known file
//!   names/segment ID/suffix/`maxDoc` rather than parsing `segments_N`/
//!   `.si` itself; see `segment.rs`'s module doc for why.
//! - **The JNI wrapper class itself** — out of scope for this Rust repo;
//!   this crate only needs to expose a stable C ABI a JNI class can bind to.
//!
//! ## Design summary (see the `ffi-safety` skill for the full rule set)
//!
//! - **Opaque handles only**: [`handle::SlotMap`] is a hand-rolled,
//!   generation-tagged `u64` slotmap (no Rust pointer/reference ever
//!   crosses the boundary); three process-wide instances
//!   ([`registry::directories`]/[`registry::segments`]/[`registry::results`])
//!   back the three handle types above.
//! - **Panics never unwind past the boundary**: every `extern "C" fn`'s
//!   body runs inside [`error::guard`], which wraps `catch_unwind` and
//!   converts a caught panic into [`error::FfiStatus::Panic`] plus a
//!   thread-local message (see `error.rs`'s module doc for the
//!   `UnwindSafe` reasoning).
//! - **Every call returns an `i32` status code**; results are delivered
//!   via out-parameters/handles, never Rust-side-allocated memory the
//!   caller must free through anything but this crate's own matching
//!   accessor (`ffi_results_copy` writes into a *caller*-allocated buffer —
//!   there is no Rust-allocated buffer handed across the boundary at all in
//!   this slice).
//! - **Every handle is validated before use**: each function looks its
//!   handle(s) up in the relevant [`handle::SlotMap`] before touching
//!   anything, returning [`error::FfiStatus::InvalidHandle`] on a miss.
//! - **No callbacks from Rust into Java**: every collector run in this
//!   crate is a plain `lucene_search::VecCollector`; the caller retrieves
//!   final results via [`results::ffi_results_copy`], never a callback.

mod directory;
mod error;
mod handle;
mod query;
mod raw;
mod registry;
mod results;
mod results_scored;
mod segment;

pub use directory::{ffi_close_directory, ffi_open_directory};
pub use error::FfiStatus;
pub use query::{
    ffi_search_boolean_query, ffi_search_boolean_query_scored, ffi_search_phrase_query,
    ffi_search_phrase_query_scored, ffi_search_term_query, ffi_search_term_query_scored,
};
pub use results::{ffi_close_results, ffi_results_copy, ffi_results_len};
pub use results_scored::{
    ffi_close_scored_results, ffi_scored_results_copy, ffi_scored_results_len,
};
pub use segment::{ffi_close_segment, ffi_open_segment};

use std::os::raw::c_char;

/// Copies the calling thread's last-error message (set by the most recent
/// non-`Ok` call on this thread) into `buf`. See [`error::get_last_error_message`]
/// for the exact contract (buffer sizing, NUL-termination, `BufferTooSmall`
/// behavior).
///
/// # Safety
/// `buf` must be valid for writes of `buf_len` bytes; `out_written` must be
/// valid for one `usize` write, or null.
#[no_mangle]
pub unsafe extern "C" fn ffi_get_last_error_message(
    buf: *mut c_char,
    buf_len: usize,
    out_written: *mut usize,
) -> i32 {
    // SAFETY: forwarding the same caller contract documented above, which
    // matches `error::get_last_error_message`'s own `# Safety` section.
    unsafe { error::get_last_error_message(buf, buf_len, out_written) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_error_message_reflects_most_recent_failure_on_this_thread() {
        // ffi_close_directory on an unknown handle sets a last-error message.
        let rc = ffi_close_directory(0xDEAD_BEEF);
        assert_eq!(rc, FfiStatus::InvalidHandle.code());

        let mut buf = [0 as c_char; 256];
        let mut written: usize = 0;
        let rc = unsafe {
            ffi_get_last_error_message(buf.as_mut_ptr(), buf.len(), &mut written as *mut _)
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert!(written > 0);
        let msg = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap();
        assert!(msg.contains("unknown or already-closed"));
    }
}
