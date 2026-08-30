//! `ffi_explain_term_query`/`ffi_explain_phrase_query`/`ffi_explain_boolean_query`
//! (Query explain FFI exposure): wraps this port's existing
//! `lucene_search::explain::explain_clause` across the FFI boundary. No
//! explain logic is reimplemented here -- every function below builds exactly
//! the same [`lucene_search::Clause`] tree `query.rs`'s matching
//! `ffi_search_*_query_scored` sibling builds (same wire formats, same
//! [`crate::query::read_term_clauses`]/norms-map helpers, reused directly, not
//! duplicated), hands it straight to `explain_clause`, and marshals the
//! resulting [`lucene_search::explain::Explanation`] tree back out. See
//! `explain.rs` (the `lucene-search` crate's, not this one) for
//! `explain_clause`'s own contract, including the bit-for-bit-identical-to-
//! `search_*_query_scored` guarantee this crate does not re-verify beyond a
//! differential test against the same fixture.
//!
//! ## Scope: only the query kinds already constructible over this C ABI
//!
//! `lucene_search::Clause` has ten variants, but this crate's existing
//! `query.rs` only ever *constructs* three of them from FFI input:
//! `Clause::Term`, `Clause::Phrase`, and (flat-`Clause::Term`-only)
//! `Clause::Boolean` -- see that module's doc comment for why nested-boolean/
//! `DisjunctionMax`/`ConstantScore`/`Boost`/`Wildcard`/`Prefix`/`Fuzzy`/
//! `Regexp`/`Span` construction isn't exposed over this ABI yet. This module
//! deliberately mirrors that exact scope rather than inventing a new, wider
//! wire format `query.rs` itself doesn't have: [`ffi_explain_term_query`]/
//! [`ffi_explain_phrase_query`]/[`ffi_explain_boolean_query`] are the explain
//! counterparts of `ffi_search_term_query_scored`/`ffi_search_phrase_query_scored`/
//! `ffi_search_boolean_query_scored` specifically, reusing their wire formats
//! verbatim. Explaining a `DisjunctionMax`/`ConstantScore`/`Boost`/flat-match
//! clause (or a truly nested `BooleanQuery`) is not reachable through this
//! crate at all yet -- not a gap introduced by this task, since none of those
//! clause shapes can be *searched* through this ABI either. See
//! `docs/parity.md` for this scope note tracked alongside the rest of this
//! crate's deferred surface.
//!
//! ## The recursive tree problem: flattening scheme
//!
//! [`lucene_search::explain::Explanation`] is a recursive tree (`details:
//! Vec<Self>`, arbitrarily deep -- e.g. a single `Clause::Term` explanation is
//! already three levels: `weight(...)` -> `score(freq=...)` ->
//! `idf`/`tfNorm`, each of the last two with their own leaf `details`). Unlike
//! `facets.rs`/`highlighter.rs`'s flat result lists (one `FacetCount`/`Fragment`
//! per element, no nesting), a tree has no natural "one element per return
//! value" shape to bulk-copy or iterate with a single index.
//!
//! This module flattens the tree into a `Vec<`[`crate::registry::ExplainNode`]`>`
//! at construction time (`flatten_explanation`, depth-first pre-order): each
//! node keeps its own `value`/`matched`/`description` plus a `Vec<usize>` of
//! its **children's indices into that same flat `Vec`** (not nested owned
//! `Explanation`s, and not a parent-index -- a child-index list per node was
//! chosen over a parent-index-per-node scheme because it makes
//! [`crate::results_explain::ffi_explain_node_child_count`]/
//! [`crate::results_explain::ffi_explain_node_child_at`]'s "give me this
//! node's Nth child" query an O(1) index into a node's own small `children`
//! list, rather than an O(total nodes) scan for every node whose
//! parent-index equals the one being queried). The root explanation is
//! always flattened first, so **index `0` is always the root** -- a caller
//! walks the whole tree by starting at node `0` and recursively following
//! `ffi_explain_node_child_at(handle, node_index, i)` for `i` in
//! `0..ffi_explain_node_child_count(handle, node_index)`, exactly the
//! "flatten with per-node accessors" scheme this task's brief describes.
//! Results are collected into a new [`crate::registry::ExplainResultsHandle`],
//! read back via `results_explain.rs`'s accessors, the same
//! new-handle-plus-accessors shape `facets.rs`/`highlighter.rs` already
//! established for their own variable-length outputs.

use std::collections::HashMap;
use std::os::raw::c_char;

use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_search::explain::{explain_clause, Explanation};
use lucene_search::field_norms::FieldNorms;
use lucene_search::{Clause, PhraseQuery, TermQuery};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::query::{check_clause_count, map_search_error, open_field_norms, read_boolean_query};
use crate::raw::{bytes_from_raw, str_from_raw, try_with_capacity};
use crate::registry::{
    explain_results, read_recovering, segments, ExplainNode, ExplainResultsHandle, SegmentHandle,
};

/// Flattens `exp` into `out` depth-first, pre-order (self pushed before any
/// child is visited), returning `exp`'s own new index in `out` -- see this
/// module's doc comment for why pre-order guarantees the root ends up at
/// index `0` for the top-level call (`out` starts empty).
fn flatten_explanation(exp: &Explanation, out: &mut Vec<ExplainNode>) -> usize {
    let idx = out.len();
    out.push(ExplainNode {
        matched: exp.matched,
        value: exp.value,
        description: exp.description.clone(),
        children: Vec::new(),
    });
    let children: Vec<usize> = exp
        .details
        .iter()
        .map(|child| flatten_explanation(child, out))
        .collect();
    out[idx].children = children;
    idx
}

/// Rejects a doc ID outside `0..max_doc` with
/// [`FfiStatus::InvalidArgument`]. `explain_clause` treats an out-of-range
/// doc as simply "not matched" and returns a no-match explanation, which
/// reads as a legitimate answer about a real document rather than the caller
/// error it is -- the same reasoning `sort.rs`'s
/// `ffi_numeric_doc_value_for_doc`/`validate_candidates` already apply to
/// caller-supplied doc IDs.
fn validate_doc(doc: i32, max_doc: i32) -> Result<(), FfiStatus> {
    if doc < 0 || doc >= max_doc {
        set_last_error(format!("explain: doc {doc} is out of range 0..{max_doc}"));
        return Err(FfiStatus::InvalidArgument);
    }
    Ok(())
}

/// Port of `IndexSearcher.explain(Weight, int)`'s deleted-document branch:
///
/// ```java
/// final Bits liveDocs = ctx.reader().getLiveDocs();
/// if (liveDocs != null && liveDocs.get(deBasedDoc) == false) {
///   return Explanation.noMatch("Document " + doc + " is deleted");
/// }
/// ```
///
/// **Why this lives here rather than falling out of `explain_clause`'s
/// `live_docs` argument.** Passing the segment's live docs down into
/// `explain_clause` does produce the right *verdict* (no match) for a deleted
/// document -- but the wrong *reason*: the doc drops out at the postings
/// lookup and the tree comes back saying `no matching term`, indistinguishable
/// from a document that simply does not contain the term. Java deliberately
/// answers this one layer up, at the same layer this module is, so the caller
/// can tell "deleted" from "absent". `ffi_explain_node_description` is exactly
/// the string an OpenSearch `_explain` response renders, so the distinction is
/// the whole point of the call.
///
/// Returns `Some(explanation)` when `doc` is deleted, `None` when it is live
/// (or the segment has no live docs attached, i.e. no deletions).
fn deleted_doc_explanation(segment: &SegmentHandle, doc: i32) -> Option<Explanation> {
    let live = segment.live_docs.as_ref()?;
    if live.get(doc as usize) {
        return None;
    }
    Some(Explanation::no_match(format!("Document {doc} is deleted")))
}

fn insert_explanation(exp: &Explanation) -> Result<u64, FfiStatus> {
    let mut nodes = Vec::new();
    flatten_explanation(exp, &mut nodes);
    explain_results().insert_checked(ExplainResultsHandle { nodes })
}

/// Explains `search_term_query_scored`'s equivalent `(field, term)` match for
/// `doc` -- wraps `lucene_search::explain::explain_clause` with
/// `Clause::Term(TermQuery::new(field, term))`, using the same
/// [`crate::query::open_field_norms`] real-norms lookup
/// `ffi_search_term_query_scored` uses. Writes a new
/// [`crate::registry::ExplainResultsHandle`] to
/// `*out_explain_results_handle` on success -- see this module's doc comment
/// for the flattened tree's shape and index-`0`-is-root convention.
///
/// # Safety
/// `field` must be valid for `field_len` bytes, `term` for `term_len` bytes,
/// `out_explain_results_handle` valid for one `u64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_explain_term_query(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    term: *const u8,
    term_len: usize,
    doc: i32,
    out_explain_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_explain_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field`/`term` are valid for their
        // paired lengths.
        let (field, term) = unsafe {
            (
                str_from_raw(field, field_len)?,
                bytes_from_raw(term, term_len)?,
            )
        };
        let query = TermQuery::new(field, term.to_vec());

        let segments = read_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_explain_term_query: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;
        validate_doc(doc, segment.max_doc)?;
        // `IndexSearcher.explain`'s deleted-document branch, before any
        // weight is consulted -- see `deleted_doc_explanation`.
        if let Some(explanation) = deleted_doc_explanation(segment, doc) {
            let handle = insert_explanation(&explanation)?;
            // SAFETY: caller contract guarantees the out-parameter is valid
            // for one write; it was null-checked at the top of this function.
            unsafe {
                *out_explain_results_handle = handle;
            }
            return Ok(());
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

        let explanation = explain_clause(
            &segment.fields,
            doc_in.as_ref(),
            None,
            None,
            segment.live_docs.as_ref(),
            &Clause::Term(query),
            doc,
            norms
                .as_ref()
                .map(|n| HashMap::from([(field.to_string(), n.clone())]))
                .as_ref(),
        )
        .map_err(map_search_error)?;

        let handle = insert_explanation(&explanation)?;
        // SAFETY: caller contract guarantees `out_explain_results_handle` is
        // valid for one write.
        unsafe {
            *out_explain_results_handle = handle;
        }
        Ok(())
    })
}

/// Explains `search_phrase_query_scored`'s equivalent multi-term-phrase match
/// for `doc` -- same single-field, in-phrase-order term list wire format as
/// [`crate::query::ffi_search_phrase_query`]/`ffi_search_phrase_query_scored`
/// (a single-term phrase needs no `.pos` file, delegating internally to the
/// same term-explain path [`ffi_explain_term_query`] uses -- see
/// `lucene_search::explain::explain_clause`'s doc comment). A multi-term
/// phrase requires the segment to have been opened with a `.pos` file
/// ([`crate::segment::ffi_open_segment`]'s `pos_name` parameter); otherwise
/// this returns [`FfiStatus::Search`], same as the unscored/scored search
/// siblings.
///
/// # Safety
/// `field` must be valid for `field_len` bytes; `terms`/`term_lens` must each
/// be valid for `term_count` elements, with every `terms[i]` valid for
/// `term_lens[i]` bytes; `out_explain_results_handle` must be valid for one
/// `u64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_explain_phrase_query(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    terms: *const *const u8,
    term_lens: *const usize,
    term_count: usize,
    doc: i32,
    out_explain_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_explain_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract guarantees `field` is valid for `field_len`
        // bytes, and (when `term_count > 0`) `terms`/`term_lens` are valid for
        // `term_count` elements with each element pair valid for its length.
        let (field, term_list) = unsafe {
            let field = str_from_raw(field, field_len)?;
            let mut term_list = try_with_capacity(term_count)?;
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

        let segments = read_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_explain_phrase_query: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;
        validate_doc(doc, segment.max_doc)?;
        // `IndexSearcher.explain`'s deleted-document branch, before any
        // weight is consulted -- see `deleted_doc_explanation`.
        if let Some(explanation) = deleted_doc_explanation(segment, doc) {
            let handle = insert_explanation(&explanation)?;
            // SAFETY: caller contract guarantees the out-parameter is valid
            // for one write; it was null-checked at the top of this function.
            unsafe {
                *out_explain_results_handle = handle;
            }
            return Ok(());
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
        let pos_in = segment
            .pos_bytes
            .as_deref()
            .map(|b| PosInput::open(b, &segment.segment_id, &segment.segment_suffix))
            .transpose()
            .map_err(|e| {
                set_last_error(format!("reopening .pos: {e}"));
                FfiStatus::Decode
            })?;
        let pay_in = segment
            .pay_bytes
            .as_deref()
            .map(|b| PayInput::open(b, &segment.segment_id, &segment.segment_suffix))
            .transpose()
            .map_err(|e| {
                set_last_error(format!("reopening .pay: {e}"));
                FfiStatus::Decode
            })?;
        let norms = open_field_norms(segment, &query.field)?;

        let explanation = explain_clause(
            &segment.fields,
            doc_in.as_ref(),
            pos_in.as_ref(),
            pay_in.as_ref(),
            segment.live_docs.as_ref(),
            &Clause::Phrase(query),
            doc,
            norms
                .as_ref()
                .map(|n| HashMap::from([(field.to_string(), n.clone())]))
                .as_ref(),
        )
        .map_err(map_search_error)?;

        let handle = insert_explanation(&explanation)?;
        // SAFETY: caller contract guarantees `out_explain_results_handle` is
        // valid for one write.
        unsafe {
            *out_explain_results_handle = handle;
        }
        Ok(())
    })
}

/// Explains `search_boolean_query_scored`'s equivalent match for `doc` --
/// same occur-tagged clause-array wire format as
/// [`crate::query::ffi_search_boolean_query`]/`ffi_search_boolean_query_scored`
/// (see that module's doc comment), and the same per-distinct-field norms map
/// construction `ffi_search_boolean_query_scored` uses.
///
/// # Safety
/// Same contract as [`crate::query::ffi_search_boolean_query`]'s, plus
/// `out_explain_results_handle` must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_explain_boolean_query(
    segment_handle: u64,
    clause_occurs: *const u8,
    clause_kinds: *const u8,
    clause_fields: *const *const c_char,
    clause_field_lens: *const usize,
    clause_terms: *const *const u8,
    clause_term_lens: *const usize,
    clause_parents: *const i32,
    clause_params: *const i32,
    clause_count: usize,
    minimum_should_match: i32,
    doc: i32,
    out_explain_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_explain_results_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // Per-query clause cap, before any decoding -- see
        // `query::check_clause_count`: one array, one length, so the
        // three-separately-capped-lists hazard cannot recur.
        check_clause_count(clause_count)?;
        // SAFETY: see `read_boolean_query`'s contract; every array/count pair
        // here matches it exactly.
        let query = unsafe {
            read_boolean_query(
                clause_occurs,
                clause_kinds,
                clause_fields,
                clause_field_lens,
                clause_terms,
                clause_term_lens,
                clause_parents,
                clause_params,
                clause_count,
                minimum_should_match,
            )?
        };

        let segments = read_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error("ffi_explain_boolean_query: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;
        validate_doc(doc, segment.max_doc)?;
        // `IndexSearcher.explain`'s deleted-document branch, before any
        // weight is consulted -- see `deleted_doc_explanation`.
        if let Some(explanation) = deleted_doc_explanation(segment, doc) {
            let handle = insert_explanation(&explanation)?;
            // SAFETY: caller contract guarantees the out-parameter is valid
            // for one write; it was null-checked at the top of this function.
            unsafe {
                *out_explain_results_handle = handle;
            }
            return Ok(());
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

        // Same per-distinct-field norms map construction as
        // `ffi_search_boolean_query_scored` (see that function's own comment) --
        // every clause here is `Clause::Term` by `read_term_clauses`'s own
        // contract, so the `Clause::Term(t) => ...` arm is the only reachable one.
        let field_names = crate::query::clause_field_names(&query);
        let mut norms_map: HashMap<String, FieldNorms<'_>> = HashMap::new();
        for name in field_names {
            if let Some(field_norms) = open_field_norms(segment, name)? {
                norms_map.insert(name.to_string(), field_norms);
            }
        }
        let norms_arg = (!norms_map.is_empty()).then_some(&norms_map);

        let explanation = explain_clause(
            &segment.fields,
            doc_in.as_ref(),
            None,
            None,
            segment.live_docs.as_ref(),
            &Clause::Boolean(Box::new(query)),
            doc,
            norms_arg,
        )
        .map_err(map_search_error)?;

        let handle = insert_explanation(&explanation)?;
        // SAFETY: caller contract guarantees `out_explain_results_handle` is
        // valid for one write.
        unsafe {
            *out_explain_results_handle = handle;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};
    use crate::results_explain::{
        ffi_close_explain_results, ffi_explain_node_child_at, ffi_explain_node_child_count,
        ffi_explain_node_description, ffi_explain_node_matched, ffi_explain_node_value,
        ffi_explain_results_len,
    };
    use crate::segment::{ffi_close_segment, ffi_open_segment};
    use lucene_search::search_term_query_scored;
    use lucene_search::ScoringCollector;

    #[derive(Default)]
    struct ScoreCapture {
        scores: Vec<(i32, f32)>,
    }
    impl ScoringCollector for ScoreCapture {
        fn collect(&mut self, doc_id: i32, score: f32) {
            self.scores.push((doc_id, score));
        }
    }

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

    /// Same fixture-opening shape as `query.rs`'s own `open_segment_with_norms`
    /// test helper (duplicated here rather than shared, matching this crate's
    /// existing per-module convention -- see `facets.rs`'s
    /// `candidates_from_raw` doc comment for the same reasoning).
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
                std::ptr::null(),
                0,
                std::ptr::null(), // nvm_name
                0,
                std::ptr::null(), // nvd_name
                0,
                std::ptr::null(), // dvm_name
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

    /// Walks the whole flattened tree rooted at node `0`, returning
    /// `(value, matched, description, child_count)` per node in the same
    /// depth-first pre-order [`flatten_explanation`] produced it in --
    /// exactly the "walk the entire tree via FFI accessors" cross-check the
    /// task brief requires, not just a root-node spot check.
    fn walk_tree(handle: u64) -> Vec<(f32, bool, String, usize)> {
        let mut len: usize = 0;
        assert_eq!(
            unsafe { ffi_explain_results_len(handle, &mut len as *mut _) },
            FfiStatus::Ok.code()
        );
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            let mut value: f32 = 0.0;
            assert_eq!(
                unsafe { ffi_explain_node_value(handle, idx, &mut value as *mut _) },
                FfiStatus::Ok.code()
            );
            let mut matched: u8 = 0;
            assert_eq!(
                unsafe { ffi_explain_node_matched(handle, idx, &mut matched as *mut _) },
                FfiStatus::Ok.code()
            );
            let mut buf = [0 as c_char; 512];
            let mut written: usize = 0;
            assert_eq!(
                unsafe {
                    ffi_explain_node_description(
                        handle,
                        idx,
                        buf.as_mut_ptr(),
                        buf.len(),
                        &mut written as *mut _,
                    )
                },
                FfiStatus::Ok.code()
            );
            let desc = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                .to_str()
                .unwrap()
                .to_string();
            let mut child_count: usize = 0;
            assert_eq!(
                unsafe { ffi_explain_node_child_count(handle, idx, &mut child_count as *mut _) },
                FfiStatus::Ok.code()
            );
            out.push((value, matched != 0, desc, child_count));
        }
        out
    }

    /// Flattens `exp` the same depth-first pre-order way
    /// [`flatten_explanation`] does, but purely in Rust (no FFI at all) --
    /// this test's own ground truth to compare [`walk_tree`]'s FFI-derived
    /// output against, so the differential check covers the *entire* tree,
    /// not just the root node's `value`.
    fn flatten_directly(exp: &Explanation) -> Vec<(f32, bool, String, usize)> {
        let mut out = Vec::new();
        fn go(exp: &Explanation, out: &mut Vec<(f32, bool, String, usize)>) {
            out.push((
                exp.value,
                exp.matched,
                exp.description.clone(),
                exp.details.len(),
            ));
            for child in &exp.details {
                go(child, out);
            }
        }
        go(exp, &mut out);
        out
    }

    /// Also verifies parent/child index linkage matches -- `walk_tree`'s flat
    /// list alone doesn't prove the `children` arrays point at the right
    /// indices, only that node *content* matches in traversal order (which
    /// coincides for `flatten_explanation`'s specific pre-order + "push
    /// children in `details` order" scheme). This directly checks
    /// `ffi_explain_node_child_at` against a Rust-side re-derivation of the
    /// same flattening.
    fn assert_child_links_match(handle: u64, exp: &Explanation) {
        fn go(handle: u64, exp: &Explanation, next_index: &mut usize) -> usize {
            let my_index = *next_index;
            *next_index += 1;
            for (i, child) in exp.details.iter().enumerate() {
                let expected_child_index_start = *next_index;
                let child_index = go(handle, child, next_index);
                assert!(child_index >= expected_child_index_start);
                let mut got: usize = 0;
                assert_eq!(
                    unsafe { ffi_explain_node_child_at(handle, my_index, i, &mut got as *mut _) },
                    FfiStatus::Ok.code()
                );
                assert_eq!(got, child_index);
            }
            my_index
        }
        let mut next_index = 0usize;
        go(handle, exp, &mut next_index);
    }

    #[test]
    fn explain_term_query_matches_direct_explain_clause_call_over_the_whole_tree() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";

        let target_doc = {
            let segs = crate::registry::read_recovering(crate::registry::segments());
            let seg = segs.get(seg_handle).unwrap();
            let doc_in = seg
                .doc_bytes
                .as_deref()
                .map(|b| {
                    lucene_codecs::postings::DocInput::open(b, &seg.segment_id, &seg.segment_suffix)
                })
                .transpose()
                .unwrap();
            let mut capture = ScoreCapture::default();
            search_term_query_scored(
                &seg.fields,
                doc_in.as_ref(),
                None,
                &TermQuery::new(field, term.to_vec()),
                None,
                &mut capture,
            )
            .unwrap();
            assert!(!capture.scores.is_empty());
            capture.scores[0].0
        };

        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                target_doc,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let got = walk_tree(out);

        // Direct call to `lucene_search::explain::explain_clause`, per this
        // task's required differential-test shape.
        let segs = crate::registry::read_recovering(crate::registry::segments());
        let seg = segs.get(seg_handle).unwrap();
        let doc_in = seg
            .doc_bytes
            .as_deref()
            .map(|b| {
                lucene_codecs::postings::DocInput::open(b, &seg.segment_id, &seg.segment_suffix)
            })
            .transpose()
            .unwrap();
        let expected = explain_clause(
            &seg.fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Term(TermQuery::new(field, term.to_vec())),
            target_doc,
            None,
        )
        .unwrap();

        assert_eq!(got, flatten_directly(&expected));
        assert!(!got.is_empty());
        assert!(got[0].1, "root must be matched for a matching doc");

        assert_child_links_match(out, &expected);
        drop(segs);

        ffi_close_explain_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_term_query_non_matching_doc_is_a_single_no_match_node() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        // Doc 1 is a real document in this segment that `body:cat` does not
        // match (the term matches docs 0 and 2 -- see `query.rs`'s
        // `term_query_matches_expected_docs`). Deliberately in range: an
        // out-of-range doc is a caller error, not a no-match answer -- see
        // `explain_rejects_a_doc_outside_the_segment` below.
        let field = "body";
        let term = b"cat";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                1,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let got = walk_tree(out);
        assert_eq!(got.len(), 1);
        assert!(!got[0].1);
        assert_eq!(got[0].0, 0.0);

        ffi_close_explain_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// A doc ID outside the segment must be `InvalidArgument`, not a
    /// plausible-looking "this document did not match" explanation of a
    /// document that does not exist.
    #[test]
    fn explain_rejects_a_doc_outside_the_segment() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let term = b"cat";

        for doc in [-1i32, 999_999] {
            let mut out: u64 = 0;
            let rc = unsafe {
                ffi_explain_term_query(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    doc,
                    &mut out as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::InvalidArgument.code(), "doc {doc}");
            assert_eq!(out, 0, "doc {doc}: no handle may be issued");
        }

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_term_query_unknown_segment_handle_is_invalid_handle() {
        let field = "body";
        let term = b"cat";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_term_query(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn explain_term_query_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let term = b"cat";
        let rc = unsafe {
            ffi_explain_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_term_query_null_field_with_nonzero_len_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let term = b"cat";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_term_query(
                seg_handle,
                std::ptr::null(),
                4,
                term.as_ptr(),
                term.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_term_query_invalid_utf8_field_is_invalid_utf8_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = [0xFFu8];
        let term = b"cat";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidUtf8.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_phrase_query_multi_term_matches_direct_explain_clause_call() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, true);

        let field = "pos";
        let terms: [&[u8]; 2] = [b"alpha", b"beta"];
        let term_ptrs: Vec<*const u8> = terms.iter().map(|t| t.as_ptr()).collect();
        let term_lens: Vec<usize> = terms.iter().map(|t| t.len()).collect();

        // "alpha beta" is known (see `lucene_search::explain`'s own test) to
        // match real doc 8555 in the "pos" field exactly, at slop 0.
        let target_doc = 8555;
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                terms.len(),
                target_doc,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let got = walk_tree(out);

        let segs = crate::registry::read_recovering(crate::registry::segments());
        let seg = segs.get(seg_handle).unwrap();
        let doc_in = seg
            .doc_bytes
            .as_deref()
            .map(|b| {
                lucene_codecs::postings::DocInput::open(b, &seg.segment_id, &seg.segment_suffix)
            })
            .transpose()
            .unwrap();
        let pos_in = seg
            .pos_bytes
            .as_deref()
            .map(|b| {
                lucene_codecs::postings::PosInput::open(b, &seg.segment_id, &seg.segment_suffix)
            })
            .transpose()
            .unwrap();
        let expected = explain_clause(
            &seg.fields,
            doc_in.as_ref(),
            pos_in.as_ref(),
            None,
            None,
            &Clause::Phrase(PhraseQuery::new(
                field,
                terms.iter().map(|t| t.to_vec()).collect::<Vec<_>>(),
            )),
            target_doc,
            None,
        )
        .unwrap();
        drop(segs);

        assert_eq!(got, flatten_directly(&expected));
        assert!(got[0].1);

        ffi_close_explain_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_phrase_query_missing_pos_input_is_search_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "pos";
        let terms: [&[u8]; 2] = [b"alpha", b"beta"];
        let term_ptrs: Vec<*const u8> = terms.iter().map(|t| t.as_ptr()).collect();
        let term_lens: Vec<usize> = terms.iter().map(|t| t.len()).collect();
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                terms.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Search.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_phrase_query_single_term_delegates_to_term_explain() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let terms: [&[u8]; 1] = [b"cat"];
        let term_ptrs: Vec<*const u8> = terms.iter().map(|t| t.as_ptr()).collect();
        let term_lens: Vec<usize> = terms.iter().map(|t| t.len()).collect();
        let mut out_phrase: u64 = 0;
        let rc = unsafe {
            ffi_explain_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                terms.len(),
                0,
                &mut out_phrase as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let phrase_tree = walk_tree(out_phrase);

        let mut out_term: u64 = 0;
        let rc = unsafe {
            ffi_explain_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                terms[0].as_ptr(),
                terms[0].len(),
                0,
                &mut out_term as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let term_tree = walk_tree(out_term);
        assert_eq!(phrase_tree, term_tree);

        ffi_close_explain_results(out_phrase);
        ffi_close_explain_results(out_term);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_phrase_query_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, true);
        let field = "pos";
        let terms: [&[u8]; 1] = [b"alpha"];
        let term_ptrs: Vec<*const u8> = terms.iter().map(|t| t.as_ptr()).collect();
        let term_lens: Vec<usize> = terms.iter().map(|t| t.len()).collect();
        let rc = unsafe {
            ffi_explain_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                terms.len(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_phrase_query_null_terms_with_nonzero_term_count_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, true);
        let field = "pos";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_phrase_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                std::ptr::null(),
                std::ptr::null(),
                2,
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_phrase_query_unknown_segment_handle_is_invalid_handle() {
        let field = "pos";
        let terms: [&[u8]; 1] = [b"alpha"];
        let term_ptrs: Vec<*const u8> = terms.iter().map(|t| t.as_ptr()).collect();
        let term_lens: Vec<usize> = terms.iter().map(|t| t.len()).collect();
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_phrase_query(
                0xFFFF,
                field.as_ptr() as *const c_char,
                field.len(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                terms.len(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn explain_boolean_query_matches_direct_explain_clause_call() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let must_field = "body";
        let must_term: &[u8] = b"cat";
        let should_field = "body";
        let should_term: &[u8] = b"dog";

        let must_fields = [must_field.as_ptr() as *const c_char];
        let must_field_lens = [must_field.len()];
        let must_terms = [must_term.as_ptr()];
        let must_term_lens = [must_term.len()];
        let should_fields = [should_field.as_ptr() as *const c_char];
        let should_field_lens = [should_field.len()];
        let should_terms = [should_term.as_ptr()];
        let should_term_lens = [should_term.len()];

        // First find a matching doc via the existing scored boolean search FFI
        // path.
        let segs = crate::registry::read_recovering(crate::registry::segments());
        let seg = segs.get(seg_handle).unwrap();
        let doc_in = seg
            .doc_bytes
            .as_deref()
            .map(|b| {
                lucene_codecs::postings::DocInput::open(b, &seg.segment_id, &seg.segment_suffix)
            })
            .transpose()
            .unwrap();
        let query = lucene_search::BooleanQuery::new()
            .with_must([TermQuery::new(must_field, must_term.to_vec())])
            .with_should([TermQuery::new(should_field, should_term.to_vec())]);
        let mut capture = ScoreCapture::default();
        lucene_search::search_boolean_query_scored(
            &seg.fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            &query,
            None,
            &mut capture,
        )
        .unwrap();
        assert!(!capture.scores.is_empty());
        let target_doc = capture.scores[0].0;
        let expected = explain_clause(
            &seg.fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            &Clause::Boolean(Box::new(query)),
            target_doc,
            None,
        )
        .unwrap();
        drop(segs);

        let mut out: u64 = 0;
        let rc = unsafe {
            crate::legacy_boolean_abi::legacy_explain_boolean_query(
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
                target_doc,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let got = walk_tree(out);
        assert_eq!(got, flatten_directly(&expected));
        assert!(got[0].1);

        assert_child_links_match(out, &expected);

        ffi_close_explain_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_boolean_query_empty_query_is_no_match() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let mut out: u64 = 0;
        let rc = unsafe {
            crate::legacy_boolean_abi::legacy_explain_boolean_query(
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
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let got = walk_tree(out);
        assert_eq!(got.len(), 1);
        assert!(!got[0].1);

        ffi_close_explain_results(out);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_boolean_query_null_out_handle_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let rc = unsafe {
            crate::legacy_boolean_abi::legacy_explain_boolean_query(
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
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_boolean_query_null_must_fields_with_nonzero_count_is_null_pointer_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let mut out: u64 = 0;
        let rc = unsafe {
            crate::legacy_boolean_abi::legacy_explain_boolean_query(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
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
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn explain_boolean_query_unknown_segment_handle_is_invalid_handle() {
        let mut out: u64 = 0;
        let rc = unsafe {
            crate::legacy_boolean_abi::legacy_explain_boolean_query(
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
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }
}

/// `IndexSearcher.explain`'s deleted-document branch, end to end against the
/// real Java-written `fixtures/data/live_docs_index/` fixture (5 docs,
/// `id:0..4`, docs 1 and 3 deleted at `del_gen == 1`).
#[cfg(test)]
mod deleted_doc_tests {
    use super::*;
    use crate::directory::{ffi_close_directory, ffi_open_directory};
    use crate::results_explain::{
        ffi_close_explain_results, ffi_explain_node_description, ffi_explain_node_matched,
    };
    use crate::segment::{ffi_close_segment, ffi_open_segment, ffi_segment_set_live_docs};

    fn open_dir() -> u64 {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/live_docs_index/"
        );
        let mut handle: u64 = 0;
        assert_eq!(
            unsafe { ffi_open_directory(path.as_ptr().cast::<c_char>(), path.len(), &mut handle) },
            FfiStatus::Ok.code()
        );
        handle
    }

    fn open_segment(dir_handle: u64) -> u64 {
        let hex = "e0811e4220a8e70d1ad3e053cc6f8ee7";
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let (fnm, tim, tip, tmd, doc) = (
            "_0.fnm",
            "_0_Lucene104_0.tim",
            "_0_Lucene104_0.tip",
            "_0_Lucene104_0.tmd",
            "_0_Lucene104_0.doc",
        );
        let suffix = "Lucene104_0";
        let mut handle: u64 = 0;
        let rc = unsafe {
            ffi_open_segment(
                dir_handle,
                fnm.as_ptr().cast::<c_char>(),
                fnm.len(),
                tim.as_ptr().cast::<c_char>(),
                tim.len(),
                tip.as_ptr().cast::<c_char>(),
                tip.len(),
                tmd.as_ptr().cast::<c_char>(),
                tmd.len(),
                doc.as_ptr().cast::<c_char>(),
                doc.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                id.as_ptr(),
                suffix.as_ptr().cast::<c_char>(),
                suffix.len(),
                5,
                &mut handle,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        handle
    }

    fn explain(seg: u64, term: &str, doc: i32) -> (bool, String) {
        let field = "id";
        let mut out: u64 = 0;
        let rc = unsafe {
            ffi_explain_term_query(
                seg,
                field.as_ptr().cast::<c_char>(),
                field.len(),
                term.as_ptr(),
                term.len(),
                doc,
                &mut out,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let mut matched: u8 = 0;
        assert_eq!(
            unsafe { ffi_explain_node_matched(out, 0, &mut matched) },
            FfiStatus::Ok.code()
        );
        let matched = matched != 0;
        let mut buf = [0 as c_char; 512];
        let mut written: usize = 0;
        assert_eq!(
            unsafe {
                ffi_explain_node_description(out, 0, buf.as_mut_ptr(), buf.len(), &mut written)
            },
            FfiStatus::Ok.code()
        );
        let description = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap()
            .to_string();
        ffi_close_explain_results(out);
        (matched, description)
    }

    /// Java answers "Document N is deleted" at the `IndexSearcher.explain`
    /// layer, *not* "no matching term" from inside the weight -- the caller
    /// must be able to tell a deleted document from one that simply lacks the
    /// term. This is exactly what pushing `live_docs` down into
    /// `explain_clause` alone would get wrong.
    #[test]
    fn a_deleted_doc_explains_as_deleted_not_as_a_missing_term() {
        let dir = open_dir();
        let seg = open_segment(dir);

        // Before live docs are attached the segment has no deletions to know
        // about, so doc 1 explains as the match it structurally is.
        let (matched, _) = explain(seg, "1", 1);
        assert!(matched, "doc 1 really does contain id:1");

        let liv = "_0_1.liv";
        assert_eq!(
            unsafe {
                ffi_segment_set_live_docs(seg, dir, liv.as_ptr().cast::<c_char>(), liv.len(), 1, 2)
            },
            FfiStatus::Ok.code()
        );

        for doc in [1i32, 3] {
            let (matched, description) = explain(seg, &doc.to_string(), doc);
            assert!(!matched, "deleted doc {doc} must not match");
            assert_eq!(
                description,
                format!("Document {doc} is deleted"),
                "must be Java's own wording, not `no matching term`"
            );
        }

        // A live document that does not contain the term still gets the
        // ordinary no-match reason -- the two cases stay distinguishable.
        let (matched, description) = explain(seg, "1", 0);
        assert!(!matched);
        assert_ne!(description, "Document 0 is deleted");

        // ...and a live matching document is unaffected.
        let (matched, _) = explain(seg, "0", 0);
        assert!(matched);

        assert_eq!(ffi_close_segment(seg), FfiStatus::Ok.code());
        ffi_close_directory(dir);
    }
}
