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
//! - [`query::ffi_search_term_query_scored_maxscore`] (MAXSCORE FFI exposure task):
//!   MAXSCORE-pruned sibling of [`query::ffi_search_term_query_scored`], single
//!   `TermQuery` only — wraps `lucene_search::search_term_query_scored_maxscore`
//!   (streams postings through a `LazyDocsCursor`, skipping whole level-0 blocks a
//!   `TopDocsCollector`'s current worst kept score proves are unreachable) instead
//!   of the eager, fully-materializing decode the other scored functions use. The
//!   only FFI entry point in this crate backed by real block-level dynamic
//!   pruning — see `query.rs`'s module doc for why `ffi_search_boolean_query_scored`
//!   has none (multi-clause `BooleanQuery` WAND/MAXSCORE pruning doesn't exist at
//!   the `lucene_search` level yet).
//! - [`results_scored::ffi_scored_results_len`]/[`results_scored::ffi_scored_results_copy`]/
//!   [`results_scored::ffi_close_scored_results`]: reads a scored results handle's
//!   `(doc_id, score)` hits back out via two caller-allocated parallel buffers (see
//!   `results_scored.rs`'s module doc for why parallel buffers, not one interleaved
//!   one), then releases it.
//! - [`sort::ffi_sort_by_doc_value`]/[`sort::ffi_sort_by_multi_valued_doc_value`]
//!   (task #40): wraps `lucene_search::sort_by_numeric_doc_value`/
//!   `lucene_search::doc_value_query::sort_by_multi_valued_doc_value`, sorting an
//!   already-known candidate doc-ID list ascending by a NUMERIC/SORTED_NUMERIC
//!   doc-value into a new [`registry::SortedResultsHandle`] -- see `sort.rs`'s
//!   module doc for the missing-value/selector wire encoding, and
//!   [`segment::ffi_open_segment`]'s `dvm_name`/`dvd_name`/`dv_suffix` parameters
//!   for how a segment's doc-values data reaches it.
//! - [`sort::ffi_numeric_doc_value_for_doc`] (FFI exposure for sparse
//!   doc-values reading): the single-doc sibling of the two entries above --
//!   wraps `lucene_codecs::doc_values::numeric_value` directly, reporting
//!   whether one doc has a value for a NUMERIC field via an `out_has_value`
//!   bool separate from the `out_value` it's only meaningful alongside, so a
//!   sparse field's "no value at all" is never collapsed into (or confused
//!   with) a stored `0`. BINARY/SORTED/SORTED_NUMERIC/SORTED_SET per-doc
//!   lookups aren't exposed yet -- see `docs/parity.md`.
//! - [`results_sorted::ffi_sorted_results_len`]/[`results_sorted::ffi_sorted_results_copy`]/
//!   [`results_sorted::ffi_close_sorted_results`]: reads a sorted results handle's
//!   `(doc_id, value)` pairs back out via two caller-allocated parallel buffers,
//!   same shape as the scored-results trio above, then releases it.
//! - [`directory_reader::ffi_open_directory_reader`]/[`directory_reader::ffi_close_directory_reader`]
//!   (task #51): opens/closes task #45's
//!   `lucene_search::directory_reader::DirectoryReader` (every segment a commit
//!   lists, opened in one call) behind its own handle/registry.
//! - [`directory_reader::ffi_search_term_query_multi_segment`]/
//!   [`directory_reader::ffi_search_boolean_query_multi_segment`] (task #51): task
//!   #41's multi-segment fan-out/merge, run against a `DirectoryReader` handle,
//!   keeping the best `top_n` globally-ranked `(doc_id, score)` hits in a
//!   [`registry::ScoredResultsHandle`] -- the same handle/registry/reader trio
//!   task #30's single-segment scored queries already use, see
//!   `directory_reader.rs`'s module doc for why no new results type was needed.
//! - [`directory_reader::ffi_search_term_query_multi_segment_concurrent`]/
//!   [`directory_reader::ffi_search_boolean_query_multi_segment_concurrent`]
//!   (Concurrent segment search FFI exposure): expose
//!   `lucene_search::multi_segment::search_term_query_multi_segment_concurrent`/
//!   `search_boolean_query_multi_segment_concurrent` (rayon-based per-segment
//!   fan-out) over the exact same wire format, handle validation, and
//!   `ScoredResultsHandle` readback the sequential wrappers above already
//!   use — no search/merge logic reimplemented, only the fan-out function
//!   called differs. See `directory_reader.rs`'s own doc comments on these
//!   two functions for the byte-for-byte-identical-to-sequential proof and
//!   the tests that exercise it directly at the FFI boundary.
//! - [`facets::ffi_facet_counts_sorted_set`]/[`facets::ffi_range_facet_counts`]
//!   (Faceted search FFI exposure): wraps `lucene_search::facets::facet_counts`/
//!   `resolve_labels`/`top_n_facets` (SortedSet string facets) and
//!   `lucene_search::facets::range_facet_counts` (NUMERIC range facets), no
//!   facet-counting logic reimplemented -- see `facets.rs`'s module doc for
//!   the wire encoding (a new [`registry::FacetResultsHandle`] for the
//!   SortedSet case, since labels are resolved from the index; direct
//!   caller-allocated output buffers for the range case, since labels there
//!   are caller-supplied input already known to the caller).
//! - [`results_facets::ffi_facet_results_len`]/[`results_facets::ffi_facet_results_copy`]/
//!   [`results_facets::ffi_facet_result_label`]/[`results_facets::ffi_close_facet_results`]:
//!   reads a facet results handle's `(ord, count)` pairs back via parallel
//!   buffers plus a per-index label accessor (see `results_facets.rs`'s
//!   module doc for why labels need their own accessor), then releases it.
//! - [`highlighter::ffi_assemble_fragments`] (Highlighter FFI exposure):
//!   wraps `lucene_search::highlighter::assemble_fragments` -- no
//!   fragment-assembly logic reimplemented -- taking the field's full text
//!   plus a caller-supplied set of `TermOffsetSpan`s (four parallel input
//!   arrays; see `highlighter.rs`'s module doc for the wire encoding) and
//!   collecting the assembled fragments into a new
//!   [`registry::FragmentResultsHandle`].
//! - [`results_fragments::ffi_fragment_results_len`]/
//!   [`results_fragments::ffi_fragment_result_text`]/
//!   [`results_fragments::ffi_fragment_result_matched_terms_len`]/
//!   [`results_fragments::ffi_fragment_result_matched_term`]/
//!   [`results_fragments::ffi_close_fragment_results`]: reads a fragment
//!   results handle's per-fragment `text` and `matched_terms` back via
//!   per-index string accessors (no fixed-size half to bulk-copy, unlike
//!   `results_facets.rs`/`results_sorted.rs` -- see `results_fragments.rs`'s
//!   module doc), then releases it.
//! - [`explain::ffi_explain_term_query`]/[`explain::ffi_explain_phrase_query`]/
//!   [`explain::ffi_explain_boolean_query`] (Query explain FFI exposure): wraps
//!   `lucene_search::explain::explain_clause` for exactly the three query
//!   kinds this crate's `query.rs` can already *construct* from FFI input
//!   (`Clause::Term`/`Clause::Phrase`/flat-`Clause::Term`-only `Clause::Boolean`)
//!   -- no explain logic reimplemented, and no wider clause-construction
//!   surface invented beyond what `query.rs` already exposes; see
//!   `explain.rs`'s module doc for that scope note and for the recursive
//!   `Explanation` tree's flattening scheme (a depth-first, pre-order
//!   `Vec<registry::ExplainNode>`, root always index `0`, each node's
//!   `children` a list of indices into that same `Vec`).
//! - [`results_explain::ffi_explain_results_len`]/[`results_explain::ffi_explain_node_value`]/
//!   [`results_explain::ffi_explain_node_matched`]/
//!   [`results_explain::ffi_explain_node_description`]/
//!   [`results_explain::ffi_explain_node_child_count`]/
//!   [`results_explain::ffi_explain_node_child_at`]/
//!   [`results_explain::ffi_close_explain_results`]: reads an explain results
//!   handle's flattened tree back out node-by-node (no fixed-size half to
//!   bulk-copy and no flat element list to walk in order, unlike every other
//!   results-handle trio in this crate -- see `results_explain.rs`'s module
//!   doc), then releases it.
//! - [`range_sort::ffi_search_numeric_range_sorted_by_field`]/
//!   [`range_sort::ffi_search_numeric_range_sorted_by_field_multi_segment`]
//!   (TopFieldCollector FFI exposure): wraps
//!   `lucene_search::doc_value_query::search_numeric_range_sorted_by_field`
//!   (single segment) and
//!   `lucene_search::multi_segment::search_numeric_range_sorted_by_field_multi_segment`
//!   (multi-segment fan-out/merge across a flat array of already-open
//!   segment handles plus caller-supplied `doc_base`s, since
//!   [`registry::DirectoryReaderHandle`] carries no doc-values data -- see
//!   `range_sort.rs`'s module doc) -- no range-matching/sort/merge logic
//!   reimplemented, results collected into the existing
//!   [`registry::SortedResultsHandle`] (same `(doc_id, value)` wire shape
//!   `sort.rs`'s functions already use, read back via the existing
//!   `results_sorted.rs` trio).
//! - [`writer::ffi_open_writer`]/[`writer::ffi_writer_add_document`]/
//!   [`writer::ffi_writer_commit`]/[`writer::ffi_writer_prepare_commit`]/
//!   [`writer::ffi_writer_finish_commit`]/[`writer::ffi_writer_rollback`]/
//!   [`writer::ffi_writer_set_merge_policy`]/[`writer::ffi_close_writer`]
//!   (IndexWriter commit/merge-policy FFI exposure): wraps
//!   `lucene_index::index_writer::IndexWriter`'s open/add_document/commit/
//!   prepare_commit/finish_commit/rollback/set_merge_policy lifecycle -- no
//!   write-side logic reimplemented, see `writer.rs`'s module doc for the
//!   wire encoding (parallel arrays for field schema/document field data,
//!   same convention `segment.rs`/`query.rs` already use) and exactly which
//!   `IndexWriter` methods/`MergePolicyConfig` knobs are and are not
//!   exposed.
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
//! - ~~**Multi-segment search / a unified `.si`-driven "open everything"
//!   entry point**~~ — closed by task #51: [`directory_reader::ffi_open_directory_reader`]
//!   parses `segments_N`/every listed segment's `.si` itself (via task #45's
//!   `DirectoryReader::open`), and
//!   [`directory_reader::ffi_search_term_query_multi_segment`]/
//!   [`directory_reader::ffi_search_boolean_query_multi_segment`] expose task
//!   #41's fan-out/merge on top of it. `ffi_open_segment` itself is unchanged
//!   (still takes already-known file names) and remains the right entry point
//!   for a caller that already has one segment's files named and wants no
//!   `DirectoryReader`-level bookkeeping at all.
//! - **`term_vectors_query::matched_term_offsets` has no FFI wrapper yet** —
//!   [`highlighter::ffi_assemble_fragments`] takes its `TermOffsetSpan`s as
//!   plain caller-supplied input (the caller computes them however it likes,
//!   e.g. by calling `lucene_search::term_vectors_query::matched_term_offsets`
//!   directly if it links against `lucene-search` itself, or by some other
//!   means); a JNI-only caller with no Rust-side access to that function
//!   would need it exposed too before it could get real spans, but that is a
//!   separate, mechanical follow-up wrapping a different `lucene-search`
//!   module, not part of this task's scope.
//! - **`explain_clause`'s `DisjunctionMax`/`ConstantScore`/`Boost`/`Wildcard`/
//!   `Prefix`/`Fuzzy`/`Regexp`/`Span`/truly-nested-`Boolean` explanations have
//!   no FFI wrapper** — none of those `Clause` variants are constructible
//!   from FFI input anywhere in this crate yet (`query.rs` only ever builds
//!   `Clause::Term`/`Clause::Phrase`/flat-`Clause::Term`-only `Clause::Boolean`
//!   from wire input), so there is nothing for an explain wrapper to explain
//!   for them either — not a gap this task introduced, since those clause
//!   shapes can't be *searched* through this ABI at all yet. Exposing them
//!   (for both search and explain) is a follow-up to `query.rs`'s own wire
//!   format, not to `explain.rs`.
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
mod directory_reader;
mod error;
mod explain;
mod facets;
mod handle;
mod highlighter;
mod points_query;
mod query;
mod range_sort;
mod raw;
mod registry;
mod results;
mod results_explain;
mod results_facets;
mod results_fragments;
mod results_scored;
mod results_sorted;
mod segment;
mod sort;
mod writer;

pub use directory::{ffi_close_directory, ffi_open_directory};
pub use directory_reader::{
    ffi_close_directory_reader, ffi_open_directory_reader, ffi_search_boolean_query_multi_segment,
    ffi_search_boolean_query_multi_segment_concurrent, ffi_search_term_query_multi_segment,
    ffi_search_term_query_multi_segment_concurrent,
};
pub use error::FfiStatus;
pub use explain::{ffi_explain_boolean_query, ffi_explain_phrase_query, ffi_explain_term_query};
pub use facets::{ffi_facet_counts_sorted_set, ffi_range_facet_counts};
pub use highlighter::ffi_assemble_fragments;
pub use points_query::ffi_search_points_range;
pub use query::{
    ffi_search_boolean_query, ffi_search_boolean_query_scored, ffi_search_phrase_query,
    ffi_search_phrase_query_scored, ffi_search_term_query, ffi_search_term_query_scored,
    ffi_search_term_query_scored_maxscore,
};
pub use range_sort::{
    ffi_search_numeric_range_sorted_by_field,
    ffi_search_numeric_range_sorted_by_field_multi_segment,
};
pub use results::{ffi_close_results, ffi_results_copy, ffi_results_len};
pub use results_explain::{
    ffi_close_explain_results, ffi_explain_node_child_at, ffi_explain_node_child_count,
    ffi_explain_node_description, ffi_explain_node_matched, ffi_explain_node_value,
    ffi_explain_results_len,
};
pub use results_facets::{
    ffi_close_facet_results, ffi_facet_result_label, ffi_facet_results_copy, ffi_facet_results_len,
};
pub use results_fragments::{
    ffi_close_fragment_results, ffi_fragment_result_matched_term,
    ffi_fragment_result_matched_terms_len, ffi_fragment_result_text, ffi_fragment_results_len,
};
pub use results_scored::{
    ffi_close_scored_results, ffi_scored_results_copy, ffi_scored_results_len,
};
pub use results_sorted::{
    ffi_close_sorted_results, ffi_sorted_results_copy, ffi_sorted_results_len,
};
pub use segment::{ffi_close_segment, ffi_open_segment};
pub use sort::{
    ffi_numeric_doc_value_for_doc, ffi_sort_by_doc_value, ffi_sort_by_multi_valued_doc_value,
};
pub use writer::{
    ffi_close_writer, ffi_open_writer, ffi_writer_add_document, ffi_writer_commit,
    ffi_writer_finish_commit, ffi_writer_pending_doc_count, ffi_writer_prepare_commit,
    ffi_writer_rollback, ffi_writer_segment_info_name, ffi_writer_segment_infos_len,
    ffi_writer_set_merge_policy,
};

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
