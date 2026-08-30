//! `ffi_search_term_query`/`ffi_search_boolean_query`/`ffi_search_phrase_query`:
//! runs this port's existing `lucene_search::search_*_query` functions
//! against an already-opened [`crate::segment::SegmentHandle`], collecting
//! every matching, live doc ID into a new [`crate::registry::ResultsHandle`]
//! entirely Rust-side (a plain [`lucene_search::VecCollector`] -- no
//! callback ever crosses back into the caller, per the `ffi-safety` skill).
//!
//! **Deletions**: every entry point here passes the segment handle's own
//! `.liv` bitset (`SegmentHandle::live_docs`, attached by
//! [`crate::segment::ffi_segment_set_live_docs`]) as `live_docs`, so a
//! deleted document is never reported as a match. A segment whose live docs
//! were never attached (or explicitly cleared) carries `None`, which is
//! `lucene_search`'s own documented "this segment has no deletions"
//! behavior -- correct for a segment with `del_gen == -1`, and the reason a
//! caller with deletions **must** call `ffi_segment_set_live_docs` after
//! opening the segment.
//!
//! **The `BooleanQuery` wire format (M2 sweep batch `c13-ffi-surface`)**: one
//! flat, `Occur`-tagged, parent-indexed clause array, decoded by
//! [`read_boolean_query`] -- see that function's doc comment for the full
//! table and for why the format is shaped this way. It replaced three
//! separate `must`/`should`/`must_not` four-array clause lists, which could
//! express neither `Occur.FILTER` (landed in `lucene_search` by
//! `c11-occur-filter`, and 44% cheaper than the equivalent `MUST`) nor a
//! nested `Clause::Boolean` (supported by `lucene_search` since task #25),
//! and which would have needed a fresh C-ABI break per additional `Occur`.
//! Under the new format an `Occur` and a clause kind are *values*, so the
//! next one of either costs no ABI change at all. `minimum_should_match`
//! (Java's `BooleanQuery.Builder.setMinimumNumberShouldMatch`) is exposed at
//! the same time, per query and per nested clause -- it had no wire
//! representation before either.
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
//! **`ffi_search_boolean_query_scored`'s clause list** is the same
//! occur-tagged clause-array wire format as the unscored
//! `ffi_search_boolean_query` above (see [`read_boolean_query`]) -- its norms
//! map is built by [`clause_field_names`], which walks the whole decoded
//! clause tree, nested `Clause::Boolean`s included, for every distinct
//! `Clause::Term` field name.

use std::collections::HashMap;
use std::os::raw::c_char;

use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_search::field_norms::FieldNorms;
use lucene_search::weight_count::{count_term_query, count_term_query_shortcut};
use lucene_search::{
    search_boolean_query, search_boolean_query_scored, search_boolean_query_scored_maxscore,
    search_phrase_query, search_phrase_query_scored, search_term_query, search_term_query_scored,
    search_term_query_scored_maxscore, search_term_query_scored_with_similarity,
};
use lucene_search::{
    BooleanQuery, Clause, PhraseQuery, ScoreDoc, TermQuery, TopDocsCollector, VecCollector,
};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw, try_with_capacity};
use crate::registry::{
    read_recovering, results, scored_results, segments, ResultsHandle, ScoredResultsHandle,
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
                str_from_raw(field, field_len)?,
                bytes_from_raw(term, term_len)?,
            )
        };
        let query = TermQuery::new(field, term.to_vec());

        let segments = read_recovering(segments());
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
            segment.live_docs.as_ref(),
            &query,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = results().insert_checked(ResultsHandle {
            docs: collector.docs,
        })?;
        // SAFETY: caller contract guarantees `out_results_handle` is valid for one write.
        unsafe {
            *out_results_handle = handle;
        }
        Ok(())
    })
}

/// Real Lucene's `IndexSearcher.getMaxClauseCount()` default (`1024`), the
/// cap `BooleanQuery.Builder.add` enforces by throwing `TooManyClauses`.
///
/// **Why the cap lives at this boundary**: this port's `BooleanQuery` has no
/// builder and no cap of its own, so nothing else in the workspace refuses an
/// arbitrarily large clause list -- and the only way one gets *built* from
/// untrusted input is here, from a caller-supplied `count`. Without it a
/// caller (or a rewritten prefix/wildcard query on the JVM side) can hand
/// over a million clauses and get a million-clause query actually executed: a
/// denial-of-service shape Java refuses outright.
pub(crate) const MAX_CLAUSE_COUNT: usize = 1024;

/// Rejects a `BooleanQuery` whose clause array is longer than
/// [`MAX_CLAUSE_COUNT`].
///
/// **Per query, not per clause list.** Java's counter lives on the
/// `BooleanQuery.Builder`, and every clause goes through the same `add`
/// regardless of its `Occur`:
/// `if (clauses.size() >= IndexSearcher.maxClauseCount) throw new TooManyClauses();`
/// (`BooleanQuery.java`). The occur-tagged wire format
/// [`read_boolean_query`] decodes makes that literal: there is one array, so
/// one length to check, and the old hazard of three separately-capped lists
/// adding up to `3 * MAX_CLAUSE_COUNT` cannot recur.
///
/// **Stricter than Java for a nested query**: Java would allow
/// [`MAX_CLAUSE_COUNT`] clauses per nesting level (each level has its own
/// `Builder`), where this counts the whole tree once. Deliberate -- the cap
/// exists here as a denial-of-service guard on a caller-supplied count, and
/// "1024 clauses total" is the bound that guard wants, not "1024 per level
/// times however many levels the caller asked for".
///
/// Java's comparison is `>=` against the size *before* the add, i.e. the
/// 1025th clause is the one that throws, so the largest accepted query has
/// exactly [`MAX_CLAUSE_COUNT`] clauses -- which is what `>` against the
/// total means here.
pub(crate) fn check_clause_count(clause_count: usize) -> Result<(), FfiStatus> {
    if clause_count > MAX_CLAUSE_COUNT {
        set_last_error(format!(
            "maxClauseCount is set to {MAX_CLAUSE_COUNT}, but this query has {clause_count} clauses"
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    Ok(())
}

/// `BooleanClause.Occur.MUST`'s ordinal in Java's own enum declaration order
/// (`MUST, FILTER, SHOULD, MUST_NOT` -- `BooleanClause.java`). The wire
/// format uses Java's ordinals rather than inventing its own numbering, so a
/// JNI caller can send `occur.ordinal()` straight through.
pub(crate) const OCCUR_MUST: u8 = 0;
/// `BooleanClause.Occur.FILTER` -- "like `MUST` except that these clauses do
/// not participate in scoring".
pub(crate) const OCCUR_FILTER: u8 = 1;
/// `BooleanClause.Occur.SHOULD`.
pub(crate) const OCCUR_SHOULD: u8 = 2;
/// `BooleanClause.Occur.MUST_NOT`.
pub(crate) const OCCUR_MUST_NOT: u8 = 3;

/// A leaf `TermQuery` clause: `clause_fields[i]`/`clause_terms[i]` are its
/// field and its raw, already-analyzed term bytes.
pub(crate) const CLAUSE_KIND_TERM: u8 = 0;
/// A nested `BooleanQuery` clause: carries no field/term of its own (both may
/// be null), and every clause naming index `i` in `clause_parents` is one of
/// its children. `clause_params[i]` is its own `minimumNumberShouldMatch`.
pub(crate) const CLAUSE_KIND_BOOLEAN: u8 = 1;

/// The deepest `clause_parents` chain this boundary accepts.
///
/// **Not a Java limit** -- real Lucene has no explicit nesting cap, only
/// `maxClauseCount`. It is a boundary-safety limit: a nested `BooleanQuery`
/// is evaluated recursively (`lucene_search::resolve_clause_docs` recurses
/// per `Clause::Boolean`), and dropping a `Box`-chained clause tree recurses
/// too, so a caller-controlled nesting depth is a caller-controlled stack
/// depth. A stack overflow is an **abort**, which `catch_unwind` cannot
/// contain (see the `ffi-safety` skill and finding 4 of `b15-ffi-core`) --
/// exactly the class of defect this crate refuses to leave reachable from a
/// caller-supplied number. 32 is far past any query a real analyzer or
/// query parser produces, and [`MAX_CLAUSE_COUNT`] independently bounds the
/// total.
pub(crate) const MAX_CLAUSE_DEPTH: usize = 32;

/// One decoded clause, before it is attached to its parent.
struct PendingClause<'a> {
    occur: u8,
    kind: u8,
    field: &'a str,
    term: &'a [u8],
    parent: i32,
}

/// Appends `clause` to `parent`'s bucket for `occur`.
fn push_clause(parent: &mut BooleanQuery, occur: u8, clause: Clause) {
    match occur {
        OCCUR_MUST => parent.must.push(clause),
        OCCUR_FILTER => parent.filter.push(clause),
        OCCUR_SHOULD => parent.should.push(clause),
        // `read_boolean_query` rejects every other tag before this runs.
        _ => parent.must_not.push(clause),
    }
}

/// Decodes a whole `BooleanQuery` from this crate's **occur-tagged clause
/// array** wire format -- the shared decoder behind every
/// `ffi_search_boolean_query*` entry point and
/// [`crate::explain::ffi_explain_boolean_query`].
///
/// # Wire format
///
/// One flat array of `clause_count` clauses, each described by the same
/// index `i` across eight parallel arrays:
///
/// | array | type | meaning |
/// |---|---|---|
/// | `clause_occurs` | `u8` | [`OCCUR_MUST`]/[`OCCUR_FILTER`]/[`OCCUR_SHOULD`]/[`OCCUR_MUST_NOT`], i.e. Java's `Occur.ordinal()` |
/// | `clause_kinds` | `u8` | [`CLAUSE_KIND_TERM`] or [`CLAUSE_KIND_BOOLEAN`] |
/// | `clause_fields`/`clause_field_lens` | `(*const c_char, usize)` | the field name, for a `TERM` clause |
/// | `clause_terms`/`clause_term_lens` | `(*const u8, usize)` | the raw term bytes, for a `TERM` clause |
/// | `clause_parents` | `i32` | index of the enclosing `BOOLEAN` clause, or `-1` for a top-level clause. Must be `< i`. May be null, meaning "every clause is top-level" |
/// | `clause_params` | `i32` | a `BOOLEAN` clause's own `minimumNumberShouldMatch`; must be `0` for a `TERM` clause. May be null, meaning "all zero" |
///
/// `minimum_should_match` is the *root* query's own
/// `minimumNumberShouldMatch` (Java's
/// `BooleanQuery.Builder.setMinimumNumberShouldMatch`).
///
/// # Why this shape, and not three more arrays
///
/// The previous wire format was three separate four-array clause lists
/// (`must_*`, `should_*`, `must_not_*`). It could not express
/// `Occur.FILTER` at all -- so a JVM caller could not build the cheaper,
/// non-scoring filter clause `lucene_search::BooleanQuery::filter` has
/// supported since the M2 sweep batch `c11-occur-filter` measured it 44%
/// cheaper than the equivalent `MUST` -- and *adding* a fourth bucket would
/// have been the second C-ABI break for the second `Occur`, with a fifth
/// waiting for whatever came next. Tagging each clause with its own `Occur`
/// makes an `Occur` a **value**, not a signature: `FILTER` cost one break
/// (this one), and any further `Occur` costs none. The same reasoning gives
/// `clause_kinds`: a new leaf clause kind that is `(field, term)`-shaped
/// (`PrefixQuery`, `WildcardQuery`, `RegexpQuery`, `FuzzyQuery`,
/// `TermInSetQuery`...) is a new tag value in an existing array, and
/// `clause_parents` makes an arbitrarily nested clause tree expressible with
/// no new arrays at all.
///
/// What would still cost an ABI change is a clause kind needing an attribute
/// this format has no room for -- a `PhraseQuery`'s ordered term *list* and
/// `f32` slop, a `BoostQuery`'s `f32` boost. `clause_params` covers the
/// integer case (it is why nested `minimumNumberShouldMatch` needed no new
/// array); an `f32` one would need a parallel `clause_float_params`. Recorded
/// here so the next reader knows exactly where the format's edge is.
///
/// # Safety
/// Every array must be valid for reads of `clause_count` elements (or null
/// where the table above allows it, and when `clause_count == 0`); each
/// `clause_fields[i]`/`clause_terms[i]` must be valid for its paired length.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn read_boolean_query(
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
) -> Result<BooleanQuery, FfiStatus> {
    if minimum_should_match < 0 {
        set_last_error(format!(
            "minimumNumberShouldMatch {minimum_should_match} is negative"
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    let mut root = BooleanQuery::new().with_minimum_should_match(minimum_should_match as usize);
    if clause_count == 0 {
        return Ok(root);
    }
    if clause_occurs.is_null() || clause_kinds.is_null() {
        return Err(FfiStatus::NullPointer);
    }
    // SAFETY: caller contract guarantees both arrays are valid for
    // `clause_count` elements.
    let (occurs, kinds) = unsafe {
        (
            std::slice::from_raw_parts(clause_occurs, clause_count),
            std::slice::from_raw_parts(clause_kinds, clause_count),
        )
    };
    // `clause_parents`/`clause_params` are optional: null means the flat,
    // all-top-level, all-default query that is by far the common case.
    let parents: Option<&[i32]> = if clause_parents.is_null() {
        None
    } else {
        // SAFETY: caller contract guarantees validity for `clause_count`.
        Some(unsafe { std::slice::from_raw_parts(clause_parents, clause_count) })
    };
    let params: Option<&[i32]> = if clause_params.is_null() {
        None
    } else {
        // SAFETY: caller contract guarantees validity for `clause_count`.
        Some(unsafe { std::slice::from_raw_parts(clause_params, clause_count) })
    };

    let mut pending: Vec<PendingClause<'_>> = try_with_capacity(clause_count)?;
    let mut depth: Vec<usize> = try_with_capacity(clause_count)?;
    for i in 0..clause_count {
        let occur = occurs[i];
        if occur > OCCUR_MUST_NOT {
            set_last_error(format!(
                "clause {i}: unknown Occur tag {occur} (expected 0=MUST, 1=FILTER, 2=SHOULD, \
                 3=MUST_NOT)"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        let kind = kinds[i];
        if kind > CLAUSE_KIND_BOOLEAN {
            set_last_error(format!(
                "clause {i}: unknown clause kind {kind} (expected 0=TERM, 1=BOOLEAN)"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        let parent = parents.map_or(-1, |p| p[i]);
        // A parent must already have been seen, which both rules out a cycle
        // (a clause can never be its own ancestor) and guarantees the
        // single reverse pass below sees every child before its parent.
        if parent < -1 || parent >= i as i32 {
            set_last_error(format!(
                "clause {i}: parent index {parent} must be -1 (top level) or an earlier clause's \
                 index"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        if parent >= 0 && kinds[parent as usize] != CLAUSE_KIND_BOOLEAN {
            set_last_error(format!(
                "clause {i}: parent clause {parent} is not a BOOLEAN clause, so it cannot contain \
                 other clauses"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        let my_depth = if parent < 0 {
            0
        } else {
            depth[parent as usize] + 1
        };
        if my_depth >= MAX_CLAUSE_DEPTH {
            set_last_error(format!(
                "clause {i}: nesting depth {} exceeds the maximum of {MAX_CLAUSE_DEPTH}",
                my_depth + 1
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        depth.push(my_depth);

        let param = params.map_or(0, |p| p[i]);
        let (field, term) = match kind {
            CLAUSE_KIND_TERM => {
                if param != 0 {
                    set_last_error(format!(
                        "clause {i}: clause_params must be 0 for a TERM clause (it is a BOOLEAN \
                         clause's minimumNumberShouldMatch), got {param}"
                    ));
                    return Err(FfiStatus::InvalidArgument);
                }
                if clause_fields.is_null()
                    || clause_field_lens.is_null()
                    || clause_terms.is_null()
                    || clause_term_lens.is_null()
                {
                    return Err(FfiStatus::NullPointer);
                }
                // SAFETY: caller contract guarantees each array is valid for
                // `clause_count` elements, and each element pair is valid for
                // its paired length.
                unsafe {
                    let field_ptr = *clause_fields.add(i);
                    let field_len = *clause_field_lens.add(i);
                    let term_ptr = *clause_terms.add(i);
                    let term_len = *clause_term_lens.add(i);
                    (
                        str_from_raw(field_ptr, field_len)?,
                        bytes_from_raw(term_ptr, term_len)?,
                    )
                }
            }
            _ => {
                if param < 0 {
                    set_last_error(format!(
                        "clause {i}: minimumNumberShouldMatch {param} is negative"
                    ));
                    return Err(FfiStatus::InvalidArgument);
                }
                ("", &[][..])
            }
        };
        pending.push(PendingClause {
            occur,
            kind,
            field,
            term,
            parent,
        });
    }

    // Every nested `BOOLEAN` clause, pre-created with its own
    // `minimumNumberShouldMatch`, so the reverse pass below can push each
    // child straight into the parent it names.
    let mut nodes: Vec<Option<BooleanQuery>> = try_with_capacity(clause_count)?;
    for (i, c) in pending.iter().enumerate() {
        nodes.push(if c.kind == CLAUSE_KIND_BOOLEAN {
            let msm = params.map_or(0, |p| p[i]) as usize;
            Some(BooleanQuery::new().with_minimum_should_match(msm))
        } else {
            None
        });
    }

    // Reverse order: a parent's index is always smaller than its children's,
    // so by the time index `p` is reached every clause that named it has
    // already been pushed into it. No recursion -- see `MAX_CLAUSE_DEPTH` for
    // why recursion over caller-controlled depth is not acceptable here.
    for i in (0..clause_count).rev() {
        let c = &pending[i];
        let clause = if c.kind == CLAUSE_KIND_BOOLEAN {
            let mut nested = nodes[i]
                .take()
                .expect("every BOOLEAN clause has a pre-created node");
            // Restore caller order: the reverse walk appended children
            // back-to-front.
            nested.must.reverse();
            nested.filter.reverse();
            nested.should.reverse();
            nested.must_not.reverse();
            Clause::Boolean(Box::new(nested))
        } else {
            Clause::Term(TermQuery::new(c.field, c.term.to_vec()))
        };
        match c.parent {
            -1 => push_clause(&mut root, c.occur, clause),
            p => push_clause(
                nodes[p as usize]
                    .as_mut()
                    .expect("a validated parent is a BOOLEAN clause with a node"),
                c.occur,
                clause,
            ),
        }
    }
    root.must.reverse();
    root.filter.reverse();
    root.should.reverse();
    root.must_not.reverse();
    Ok(root)
}

/// Every distinct field name a decoded `BooleanQuery`'s clauses mention, at
/// any nesting depth -- the set a scored search needs norms for. Iterative
/// (an explicit stack), for the same caller-controlled-depth reason
/// [`MAX_CLAUSE_DEPTH`] gives.
pub(crate) fn clause_field_names(query: &BooleanQuery) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut stack: Vec<&BooleanQuery> = vec![query];
    while let Some(q) = stack.pop() {
        for clause in q
            .must
            .iter()
            .chain(q.filter.iter())
            .chain(q.should.iter())
            .chain(q.must_not.iter())
        {
            match clause {
                Clause::Term(t) => {
                    if !out.contains(&t.field.as_str()) {
                        out.push(t.field.as_str());
                    }
                }
                Clause::Boolean(nested) => stack.push(nested),
                // `read_boolean_query` builds only `Term` and `Boolean`.
                _ => {}
            }
        }
    }
    out
}

/// Runs `search_boolean_query` against `segment_handle`. The query is passed
/// as one flat, `Occur`-tagged, parent-indexed clause array -- see
/// [`read_boolean_query`] for the full wire format, including `Occur.FILTER`
/// and nested `BOOLEAN` clauses. A clause-less query is `clause_count == 0`
/// with every array pointer allowed to be null.
///
/// # Safety
/// Every `(pointer, len)` / `(array, count)` pair must be valid for the
/// documented reads; `out_results_handle` must be valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_boolean_query(
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
    out_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_results_handle.is_null() {
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
            segment.live_docs.as_ref(),
            None,
            &query,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = results().insert_checked(ResultsHandle {
            docs: collector.docs,
        })?;
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
        let pay_in = segment
            .pay_bytes
            .as_deref()
            .map(|b| PayInput::open(b, &segment.segment_id, &segment.segment_suffix))
            .transpose()
            .map_err(|e| {
                set_last_error(format!("reopening .pay: {e}"));
                FfiStatus::Decode
            })?;

        let mut collector = VecCollector::default();
        search_phrase_query(
            &segment.fields,
            doc_in.as_ref(),
            pos_in.as_ref(),
            pay_in.as_ref(),
            segment.live_docs.as_ref(),
            &query,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = results().insert_checked(ResultsHandle {
            docs: collector.docs,
        })?;
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
/// **`avgFieldLength` comes from the field's `.tmd` stats, not from
/// averaging decoded norms** (M2 sweep b15, closing the b12/b13 carry-over
/// that named this function as the last production caller of the wrong
/// one): [`FieldNorms::from_field_stats`] is Java's
/// `sumTotalTermFreq / docCount` exactly, while [`FieldNorms::open`] decodes
/// each doc's `SmallFloat`-quantized norm and averages *those* -- the
/// average of the lossy values sits 0.1-0.6% off the average of the true
/// lengths, enough to reorder documents at the top-k boundary. It is also
/// O(1) rather than O(maxDoc) per query, so the "recomputed on every call,
/// not cached" note this doc used to carry no longer has anything to
/// justify: two integer reads per call is not worth a cache.
///
/// Deletions are deliberately *not* subtracted here. Java's `docCount` (like
/// this `.tmd` counter) includes deleted documents, so `avgdl` is unaffected
/// by them; deletions are applied where Lucene applies them, to the matched
/// doc set, via each `search_*` call's `live_docs` argument (see
/// [`crate::segment::ffi_segment_set_live_docs`]).
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
    // `from_field_stats`, not `open`: real Lucene's `BM25Similarity` takes
    // `avgdl = sumTotalTermFreq / docCount` straight from the field's `.tmd`
    // aggregate counters. `FieldNorms::open` instead decodes each doc's norm
    // and averages *those*, and norms are `SmallFloat`-quantized into one
    // byte -- the average of the lossy values is systematically 0.1-0.6% off
    // the average of the true lengths, which is enough to reorder documents
    // at the top-k boundary (M1's benchmark cross-check: 19 of 20 queries
    // disagreed with Java on hit sets for this reason alone, see
    // `docs/benchmarks/verdict.md`). It is also O(1) instead of O(maxDoc)
    // per query. This was the b12/b13 carry-over naming `lucene-ffi` as the
    // remaining production caller of the wrong one.
    //
    // The stats come from this segment's own `.tmd`, and `doc_count` there
    // counts deleted documents -- which is exactly what Java's `docCount`
    // does too, so this stays correct now that deletions are honoured
    // elsewhere (see `ffi_segment_set_live_docs`).
    let Some(field_terms) = segment.fields.field(field) else {
        return Ok(None);
    };
    Ok(Some(FieldNorms::from_field_stats(
        data,
        *entry,
        field_terms.sum_total_term_freq,
        field_terms.doc_count,
    )))
}

/// `IndexSearcher.count(new TermQuery(field, term))` for one segment, written
/// to `*out_count`.
///
/// **Why this is not `ffi_search_term_query` plus a length.** Java's
/// `TotalHitCountCollector` asks `Weight.count(context)` before it opens
/// anything, and `TermWeight.count` answers from the terms dictionary's own
/// `docFreq` whenever the segment has no deletions -- no `.doc` file, no block
/// decode, no per-document loop. This entry point is that shortcut; a caller
/// that wanted a count and got a results handle paid for the postings walk it
/// then threw away. See
/// [`lucene_search::weight_count::count_term_query`] for the fallback when the
/// segment *does* have deletions, where the docFreq counts documents that are
/// no longer live and a scan is the only correct answer.
///
/// A field or term that is not in this segment counts `0`, as Java's "the term
/// cannot be found in the dictionary so the count is 0" does -- not an error.
///
/// # Safety
/// `field` must be valid for `field_len` bytes, `term` for `term_len` bytes,
/// `out_count` valid for one `i64` write.
#[no_mangle]
pub unsafe extern "C" fn ffi_count_term_query(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    term: *const u8,
    term_len: usize,
    out_count: *mut i64,
) -> i32 {
    guard(|| {
        if out_count.is_null() {
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
            set_last_error("ffi_count_term_query: unknown or already-closed segment handle");
            FfiStatus::InvalidHandle
        })?;

        // The whole point: with no deletions the answer is in the terms
        // dictionary, so `.doc` is never opened.
        let count =
            match count_term_query_shortcut(&segment.fields, segment.live_docs.as_ref(), &query) {
                Some(n) => n,
                None => {
                    let doc_in = segment
                        .doc_bytes
                        .as_deref()
                        .map(|b| DocInput::open(b, &segment.segment_id, &segment.segment_suffix))
                        .transpose()
                        .map_err(|e| {
                            set_last_error(format!("reopening .doc: {e}"));
                            FfiStatus::Decode
                        })?;
                    count_term_query(
                        &segment.fields,
                        doc_in.as_ref(),
                        segment.live_docs.as_ref(),
                        &query,
                    )
                    .map_err(map_search_error)?
                }
            };
        // SAFETY: caller contract guarantees `out_count` is valid for one write.
        unsafe {
            *out_count = count;
        }
        Ok(())
    })
}

/// [`ffi_search_term_query_scored`]'s paginating sibling:
/// `IndexSearcher.searchAfter(new ScoreDoc(after_doc, after_score), query,
/// top_n)`, i.e. the page that follows the one ending at
/// `(after_doc, after_score)`.
///
/// `after_doc`/`after_score` must be a hit a previous call returned, unmodified
/// -- the boundary is "ranks at or below this exact `(score, doc)` pair" under
/// `HitQueue`'s order, so a rounded score or a doc id from a different query
/// moves the page. This entry point is single-segment, so the doc id needs no
/// `docBase` translation; the multi-segment equivalent
/// (`lucene_search::multi_segment::search_term_query_multi_segment_after`) does
/// it internally.
///
/// Kept as its own function rather than as extra parameters on
/// [`ffi_search_term_query_scored`]: there is no `after` value that means "no
/// `after`" (`(NO_MORE_DOCS, +inf)` would, but only by convention), and a
/// sentinel a caller can get subtly wrong is exactly how a paginating caller
/// silently re-reads page 1.
///
/// # Safety
/// `field` must be valid for `field_len` bytes, `term` for `term_len` bytes,
/// `out_scored_results_handle` valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_term_query_scored_after(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    term: *const u8,
    term_len: usize,
    top_n: usize,
    after_doc: i32,
    after_score: f32,
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
                str_from_raw(field, field_len)?,
                bytes_from_raw(term, term_len)?,
            )
        };
        let query = TermQuery::new(field, term.to_vec());

        let segments = read_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_term_query_scored_after: unknown or already-closed segment handle",
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

        let mut collector = TopDocsCollector::new(top_n).with_after(ScoreDoc {
            doc_id: after_doc,
            score: after_score,
        });
        search_term_query_scored(
            &segment.fields,
            doc_in.as_ref(),
            segment.live_docs.as_ref(),
            &query,
            norms.as_ref(),
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = scored_results().insert_checked(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        })?;
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
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
                str_from_raw(field, field_len)?,
                bytes_from_raw(term, term_len)?,
            )
        };
        let query = TermQuery::new(field, term.to_vec());

        let segments = read_recovering(segments());
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
            segment.live_docs.as_ref(),
            &query,
            norms.as_ref(),
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = scored_results().insert_checked(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        })?;
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// [`ffi_search_term_query_scored`]'s sibling taking explicit `k1`/`b`
/// (task #214, "Configurable BM25 constant from FFI") instead of always
/// using [`lucene_search::similarity::DEFAULT_K1`]/
/// [`lucene_search::similarity::DEFAULT_B`] -- the one new FFI entry point
/// this task adds, for the single most fundamental scored search path (a
/// plain `TermQuery`, no MAXSCORE pruning). Runs
/// [`lucene_search::search_term_query_scored_with_similarity`], otherwise
/// identical to `ffi_search_term_query_scored`'s contract (same
/// `(field, term)`/`top_n`/[`ScoredResultsHandle`] handling, same norms
/// lookup via [`open_field_norms`]).
///
/// `k1 == lucene_search::similarity::DEFAULT_K1` and
/// `b == lucene_search::similarity::DEFAULT_B` reproduce
/// `ffi_search_term_query_scored`'s scores byte-for-byte (same underlying
/// formula, same constants) -- see
/// `ffi_search_term_query_scored_with_similarity_using_defaults_matches_hardcoded_path`
/// below for the regression proof.
///
/// **`k1`/`b` are validated, not trusted** (M2 sweep b15): they go through
/// [`lucene_search::similarity::Bm25Params::new`], real Lucene's
/// `BM25Similarity(float k1, float b)` constructor checks -- `k1` must be
/// finite and `>= 0`, `b` must be in `0..=1`. Anything else is
/// [`FfiStatus::InvalidArgument`] carrying Lucene's own verbatim message
/// (retrievable via [`crate::ffi_get_last_error_message`]), the C-ABI
/// equivalent of Java's `IllegalArgumentException`. This is not cosmetic:
/// a `b` outside `0..=1` makes BM25's length normalization non-monotonic in
/// the norm, which invalidates the impacts-derived score bounds this crate's
/// MAXSCORE paths use, so accepting one would silently *drop matching
/// documents* rather than merely score them oddly.
///
/// **Scope note** (see `lucene_search::similarity::Bm25Params`'s doc comment
/// and `docs/parity.md`'s BM25/similarity row for the full, honest list):
/// only this function and its Rust-level counterpart
/// (`search_term_query_scored_with_similarity`) support configurable `k1`/`b`
/// today. `ffi_search_term_query_scored_maxscore`,
/// `ffi_search_boolean_query_scored`/`ffi_search_boolean_query_scored_maxscore`,
/// and `ffi_search_phrase_query_scored` are all deliberately left hardcoded
/// to the BM25 defaults -- threading custom `k1`/`b` through MAXSCORE
/// pruning, multi-segment fan-out, and phrase scoring is a materially larger,
/// riskier change than this task's scope.
///
/// # Safety
/// `field` must be valid for `field_len` bytes, `term` for `term_len` bytes,
/// `out_scored_results_handle` valid for one `u64` write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_search_term_query_scored_with_similarity(
    segment_handle: u64,
    field: *const c_char,
    field_len: usize,
    term: *const u8,
    term_len: usize,
    k1: f32,
    b: f32,
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
                str_from_raw(field, field_len)?,
                bytes_from_raw(term, term_len)?,
            )
        };
        let query = TermQuery::new(field, term.to_vec());
        // `Bm25Params::new` is `BM25Similarity(float k1, float b)`'s
        // validating constructor, and the validation is load-bearing here
        // rather than decorative: `b` outside `0..=1` makes the
        // length-normalization term non-monotonic in the norm, which
        // invalidates the impacts-derived upper bounds MAXSCORE block
        // skipping relies on (missing hits, not just odd scores), and a
        // negative/non-finite `k1` produces infinities. These two floats come
        // straight off the C ABI from a JVM caller, so this is exactly the
        // "a caller (including the FFI one) could set any float" case that
        // constructor exists to stop -- surfaced as
        // `FfiStatus::InvalidArgument` plus Lucene's own verbatim message,
        // the same way Java throws `IllegalArgumentException`.
        let params = lucene_search::similarity::Bm25Params::new(k1, b).map_err(|message| {
            set_last_error(format!(
                "ffi_search_term_query_scored_with_similarity: {message}"
            ));
            FfiStatus::InvalidArgument
        })?;

        let segments = read_recovering(segments());
        let segment = segments.get(segment_handle).ok_or_else(|| {
            set_last_error(
                "ffi_search_term_query_scored_with_similarity: unknown or already-closed segment handle",
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
        search_term_query_scored_with_similarity(
            &segment.fields,
            doc_in.as_ref(),
            segment.live_docs.as_ref(),
            &query,
            norms.as_ref(),
            params,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = scored_results().insert_checked(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        })?;
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
                str_from_raw(field, field_len)?,
                bytes_from_raw(term, term_len)?,
            )
        };
        let query = TermQuery::new(field, term.to_vec());

        let segments = read_recovering(segments());
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
            segment.live_docs.as_ref(),
            &query,
            norms.as_ref(),
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = scored_results().insert_checked(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        })?;
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// Scored sibling of [`ffi_search_boolean_query`]: same occur-tagged
/// clause-array wire format (see [`read_boolean_query`]), but
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
    top_n: usize,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
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
        let field_names = clause_field_names(&query);
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
            segment.live_docs.as_ref(),
            None,
            &query,
            norms_arg,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = scored_results().insert_checked(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        })?;
        // SAFETY: caller contract guarantees `out_scored_results_handle` is valid
        // for one write.
        unsafe {
            *out_scored_results_handle = handle;
        }
        Ok(())
    })
}

/// MAXSCORE-pruned sibling of [`ffi_search_boolean_query_scored`]: same flat,
/// occur-tagged clause-array wire format and the exact same
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
    top_n: usize,
    out_scored_results_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_scored_results_handle.is_null() {
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
        let field_names = clause_field_names(&query);
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
            segment.live_docs.as_ref(),
            None,
            &query,
            norms_arg,
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = scored_results().insert_checked(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        })?;
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

        let mut collector = TopDocsCollector::new(top_n);
        search_phrase_query_scored(
            &segment.fields,
            doc_in.as_ref(),
            pos_in.as_ref(),
            pay_in.as_ref(),
            segment.live_docs.as_ref(),
            &query,
            norms.as_ref(),
            &mut collector,
        )
        .map_err(map_search_error)?;

        let handle = scored_results().insert_checked(ScoredResultsHandle {
            hits: collector.top_docs().to_vec(),
        })?;
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
                std::ptr::null(),
                0,
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

    /// The calling thread's last-error message, read back through the real
    /// exported accessor so these tests also prove it reaches a JNI caller.
    fn last_error_message() -> String {
        let mut buf = [0 as c_char; 512];
        let rc = unsafe {
            crate::ffi_get_last_error_message(buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
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

    fn count_term(seg_handle: u64, field: &str, term: &[u8]) -> (i32, i64) {
        let mut count: i64 = -7;
        let rc = unsafe {
            ffi_count_term_query(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                &mut count as *mut _,
            )
        };
        (rc, count)
    }

    /// `Weight.count`'s shortcut over the C ABI. The segment is opened
    /// **without** `.doc`, so the count can only come from the terms
    /// dictionary -- which is the entire claim.
    #[test]
    fn count_term_query_answers_from_the_terms_dictionary_without_a_doc_file() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        // `body:cat` is in 2 of the fixture's documents, `body:dog` in 2,
        // `big:everywhere` in 300 -- all straight off `docFreq`.
        assert_eq!(
            count_term(seg_handle, "body", b"cat"),
            (FfiStatus::Ok.code(), 2)
        );
        assert_eq!(
            count_term(seg_handle, "big", b"everywhere"),
            (FfiStatus::Ok.code(), 300)
        );
        // A missing term, and a missing field, both count 0 rather than
        // erroring -- Java's "the term cannot be found in the dictionary".
        assert_eq!(
            count_term(seg_handle, "body", b"zzz-missing"),
            (FfiStatus::Ok.code(), 0)
        );
        assert_eq!(
            count_term(seg_handle, "no-such-field", b"cat"),
            (FfiStatus::Ok.code(), 0)
        );
        // And it agrees with actually running the query.
        let mut results_handle: u64 = 0;
        let field = "body";
        let term = b"cat";
        assert_eq!(
            unsafe {
                ffi_search_term_query(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    &mut results_handle as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );
        assert_eq!(read_results(results_handle).len(), 2);

        ffi_close_results(results_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// The deletions half: with a `.liv` attached the `docFreq` shortcut must
    /// stand down, because `docFreq` counts documents that are no longer live.
    /// `live_docs_index` deletes documents 1 and 3, so `id:1` has `docFreq == 1`
    /// and a true count of 0 -- the number real Lucene's
    /// `IndexSearcher.count` returns (`count.term.id.1=0` in that fixture's
    /// manifest).
    ///
    /// Its own fixture directory, opened here rather than shared: the
    /// blocktree fixture this module's other tests use has no deletions at all.
    /// (Same per-module duplication `explain.rs`'s own `open_segment` helper
    /// documents.)
    #[test]
    fn count_term_query_does_not_take_the_shortcut_when_the_segment_has_deletions() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/live_docs_index/"
        );
        let mut dir_handle: u64 = 0;
        assert_eq!(
            unsafe {
                crate::directory::ffi_open_directory(
                    path.as_ptr().cast::<c_char>(),
                    path.len(),
                    &mut dir_handle,
                )
            },
            FfiStatus::Ok.code()
        );
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
        let mut seg_handle: u64 = 0;
        let rc = unsafe {
            crate::segment::ffi_open_segment(
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
                &mut seg_handle,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        // Before the `.liv` is attached the segment looks deletion-free, so the
        // shortcut answers `docFreq` -- 1 -- which is the wrong answer for a
        // reader that has the deletions.
        assert_eq!(
            count_term(seg_handle, "id", b"1"),
            (FfiStatus::Ok.code(), 1)
        );

        let liv = "_0_1.liv";
        assert_eq!(
            unsafe {
                crate::segment::ffi_segment_set_live_docs(
                    seg_handle,
                    dir_handle,
                    liv.as_ptr().cast::<c_char>(),
                    liv.len(),
                    1,
                    2,
                )
            },
            FfiStatus::Ok.code()
        );

        // Real Lucene: `count.term.id.1=0`, `count.term.id.0=1`.
        assert_eq!(
            count_term(seg_handle, "id", b"1"),
            (FfiStatus::Ok.code(), 0)
        );
        assert_eq!(
            count_term(seg_handle, "id", b"0"),
            (FfiStatus::Ok.code(), 1)
        );

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// `searchAfter` over the C ABI: three pages that partition the ranking.
    #[test]
    fn search_term_query_scored_after_pages_the_ranking_without_repeats() {
        let dir_handle = open_dir();
        let seg_handle = open_segment_with_norms(dir_handle, false, true);
        let field = "big";
        let term = b"everywhere";

        let mut page1: u64 = 0;
        assert_eq!(
            unsafe {
                ffi_search_term_query_scored(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    3,
                    &mut page1 as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );
        let hits1 = read_scored_results(page1);
        assert_eq!(hits1.len(), 3);

        let (after_doc, after_score) = *hits1.last().expect("a full page");
        let mut page2: u64 = 0;
        assert_eq!(
            unsafe {
                ffi_search_term_query_scored_after(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    3,
                    after_doc,
                    after_score,
                    &mut page2 as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );
        let hits2 = read_scored_results(page2);
        assert_eq!(hits2.len(), 3);
        for (doc, _) in &hits2 {
            assert!(
                !hits1.iter().any(|(d, _)| d == doc),
                "doc {doc} appeared on both pages"
            );
        }

        // The two pages together must be the top 6, in order.
        let mut six: u64 = 0;
        assert_eq!(
            unsafe {
                ffi_search_term_query_scored(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    6,
                    &mut six as *mut _,
                )
            },
            FfiStatus::Ok.code()
        );
        let expected = read_scored_results(six);
        let walked: Vec<(i32, f32)> = hits1.iter().chain(hits2.iter()).copied().collect();
        assert_eq!(walked, expected);

        // A null out-pointer is still rejected before anything is opened.
        assert_eq!(
            unsafe {
                ffi_search_term_query_scored_after(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    3,
                    after_doc,
                    after_score,
                    std::ptr::null_mut(),
                )
            },
            FfiStatus::NullPointer.code()
        );
        assert_eq!(
            unsafe {
                ffi_search_term_query_scored_after(
                    0xFFFF,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    3,
                    after_doc,
                    after_score,
                    &mut page2 as *mut _,
                )
            },
            FfiStatus::InvalidHandle.code()
        );

        ffi_close_scored_results(page1);
        ffi_close_scored_results(page2);
        ffi_close_scored_results(six);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn count_term_query_rejects_a_bad_handle_and_a_null_out_pointer() {
        assert_eq!(
            count_term(0xFFFF, "body", b"cat").0,
            FfiStatus::InvalidHandle.code()
        );
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let term = b"cat";
        let rc = unsafe {
            ffi_count_term_query(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query(
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
        let mut segments = crate::registry::lock_recovering(segments());
        let segment = segments.get_mut(seg_handle).expect("segment handle");
        segment.doc_bytes = Some(vec![0u8; 4]);
    }

    /// Same idea as [`corrupt_doc_bytes`], for the `.pos` file.
    fn corrupt_pos_bytes(seg_handle: u64) {
        let mut segments = crate::registry::lock_recovering(segments());
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
            crate::legacy_boolean_abi::legacy_search_boolean_query(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query(
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

    /// Regression test for the poisoning fix in `registry::lock_recovering`/
    /// `registry::read_recovering`:
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
        // Since the M2 sweep this registry is an `RwLock` and a query holds
        // only its *read* guard, which `std` never poisons (a shared borrow
        // cannot leave the map half-written). The property under test is
        // unchanged and still worth pinning: whether or not the lock ends up
        // poisoned, a later call must not be wedged by the panic.
        let _ = segments().is_poisoned();

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

    /// Java's `BooleanQuery.Builder.add` throws `TooManyClauses` past
    /// `IndexSearcher.getMaxClauseCount()` (1024). Nothing else in this port
    /// caps a clause list, and this boundary is the only place one gets built
    /// from untrusted input, so the cap lives here -- as a status code, not a
    /// panic, and not a million-clause query actually executed.
    #[test]
    fn boolean_query_rejects_more_clauses_than_lucenes_max_clause_count() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, true);

        let field = "body";
        let term = b"cat";
        let over = crate::query::MAX_CLAUSE_COUNT + 1;
        let field_ptrs: Vec<*const c_char> = vec![field.as_ptr() as *const c_char; over];
        let field_lens: Vec<usize> = vec![field.len(); over];
        let term_ptrs: Vec<*const u8> = vec![term.as_ptr(); over];
        let term_lens: Vec<usize> = vec![term.len(); over];

        let mut out: u64 = 0;
        let rc = unsafe {
            crate::legacy_boolean_abi::legacy_search_boolean_query(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                field_ptrs.as_ptr(),
                field_lens.as_ptr(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                over,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert_eq!(out, 0, "no results handle may be issued");

        let mut buf = [0 as c_char; 256];
        assert_eq!(
            unsafe {
                crate::ffi_get_last_error_message(buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
            },
            FfiStatus::Ok.code()
        );
        let msg = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap();
        assert!(msg.contains("maxClauseCount"), "message was {msg:?}");

        // The cap is per *query*, not per clause list: splitting the same
        // 1025 clauses across `must`/`should`/`must_not` must not slip past
        // it (three per-list checks would have accepted 3 * 1024).
        let third = crate::query::MAX_CLAUSE_COUNT / 2;
        let mut out: u64 = 0;
        let rc = unsafe {
            crate::legacy_boolean_abi::legacy_search_boolean_query(
                seg_handle,
                field_ptrs.as_ptr(),
                field_lens.as_ptr(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                third,
                field_ptrs.as_ptr(),
                field_lens.as_ptr(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                third,
                field_ptrs.as_ptr(),
                field_lens.as_ptr(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                third,
                &mut out as *mut _,
            )
        };
        assert_eq!(
            rc,
            FfiStatus::InvalidArgument.code(),
            "3 x 512 = 1536 clauses is over Java's 1024 ceiling"
        );
        assert_eq!(out, 0);

        // Exactly at the cap (summed) is still accepted: Java's `>=`-against-
        // the-pre-add-size check makes the 1025th clause the one that throws,
        // so a 1024-clause query is legal.
        let at = crate::query::MAX_CLAUSE_COUNT;
        let mut out: u64 = 0;
        let rc = unsafe {
            crate::legacy_boolean_abi::legacy_search_boolean_query(
                seg_handle,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                field_ptrs.as_ptr(),
                field_lens.as_ptr(),
                term_ptrs.as_ptr(),
                term_lens.as_ptr(),
                at,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut out as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        ffi_close_results(out);

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Every `k1`/`b` real Lucene's `BM25Similarity` constructor throws on
    /// must be an `FfiStatus::InvalidArgument` here with a retrievable
    /// message -- never silently accepted (which would corrupt MAXSCORE's
    /// score bounds) and never a panic.
    #[test]
    fn term_query_scored_with_similarity_rejects_out_of_range_bm25_parameters() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, true);
        let field = "body";
        let term = b"cat";

        for (k1, b, why) in [
            (-0.1f32, 0.75f32, "negative k1"),
            (f32::NAN, 0.75, "NaN k1"),
            (f32::INFINITY, 0.75, "infinite k1"),
            (1.2, -0.01, "b below 0"),
            (1.2, 1.01, "b above 1"),
            (1.2, f32::NAN, "NaN b"),
        ] {
            let mut handle: u64 = 0;
            let rc = unsafe {
                ffi_search_term_query_scored_with_similarity(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    k1,
                    b,
                    10,
                    &mut handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::InvalidArgument.code(), "{why}");
            assert_eq!(handle, 0, "{why}: no results handle may be issued");

            let mut buf = [0 as c_char; 256];
            let rc = unsafe {
                crate::ffi_get_last_error_message(buf.as_mut_ptr(), buf.len(), std::ptr::null_mut())
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            let msg = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
                .to_str()
                .unwrap();
            assert!(
                msg.contains("illegal k1 value") || msg.contains("illegal b value"),
                "{why}: message was {msg:?}"
            );
        }

        // The boundary values Lucene *accepts* must still work.
        for (k1, b) in [(0.0f32, 0.0f32), (0.0, 1.0), (1.2, 0.75)] {
            let mut handle: u64 = 0;
            let rc = unsafe {
                ffi_search_term_query_scored_with_similarity(
                    seg_handle,
                    field.as_ptr() as *const c_char,
                    field.len(),
                    term.as_ptr(),
                    term.len(),
                    k1,
                    b,
                    10,
                    &mut handle as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code(), "k1={k1} b={b}");
            crate::results_scored::ffi_close_scored_results(handle);
        }

        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_scored_with_similarity_using_defaults_matches_hardcoded_path() {
        // Task #214 regression proof, at the FFI boundary: k1/b ==
        // DEFAULT_K1/DEFAULT_B through the new entry point must reproduce
        // `ffi_search_term_query_scored`'s scores byte-for-byte.
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";

        let mut default_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                10,
                &mut default_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let mut with_similarity_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored_with_similarity(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                lucene_search::similarity::DEFAULT_K1,
                lucene_search::similarity::DEFAULT_B,
                10,
                &mut with_similarity_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let default_hits = read_scored_results(default_handle);
        let with_similarity_hits = read_scored_results(with_similarity_handle);
        assert_eq!(default_hits, with_similarity_hits);
        assert!(!default_hits.is_empty());

        ffi_close_scored_results(default_handle);
        ffi_close_scored_results(with_similarity_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_scored_with_similarity_using_different_k1_b_matches_hand_computed_value() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";
        let k1 = 2.0f32;
        let b = 0.5f32;
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored_with_similarity(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                k1,
                b,
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());

        let hits = read_scored_results(scored_handle);
        // Same fixture facts as `term_query_scored_body_cat_returns_expected_docs_and_scores`
        // (cat: docFreq=2, body.docCount=4; doc 0 freq=2, doc 2 freq=1), but with
        // k1=2.0, b=0.5 instead of the 1.2/0.75 defaults:
        // idf(2,4) = ln(1 + (4-2+0.5)/(2+0.5)) = ln(1 + 2.5/2.5) = ln(2) = 0.693147...
        // tfNorm(freq, 1, 1, 2.0, 0.5) = freq / (freq + 2.0*(1-0.5+0.5*1/1))
        //                              = freq / (freq + 2.0*1.0) = freq / (freq + 2.0)
        // doc 0 (freq=2): tfNorm = 2/4 = 0.5, score = 0.693147 * 0.5 = 0.346574...
        // doc 2 (freq=1): tfNorm = 1/3 = 0.333333..., score = 0.693147 * 0.333333... = 0.231049...
        let idf = lucene_search::similarity::idf(2, 4);
        let expected_doc0 = idf
            * lucene_search::similarity::tf_norm(
                2.0,
                lucene_search::similarity::UNNORMED_FIELD_LENGTH,
                lucene_search::similarity::UNNORMED_FIELD_LENGTH,
                k1,
                b,
            );
        let expected_doc2 = idf
            * lucene_search::similarity::tf_norm(
                1.0,
                lucene_search::similarity::UNNORMED_FIELD_LENGTH,
                lucene_search::similarity::UNNORMED_FIELD_LENGTH,
                k1,
                b,
            );
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 0);
        assert!(
            (hits[0].1 - expected_doc0).abs() < 1e-4,
            "got {}",
            hits[0].1
        );
        assert!(
            (expected_doc0 - 0.346_574).abs() < 1e-3,
            "hand-computed sanity check: {expected_doc0}"
        );
        assert_eq!(hits[1].0, 2);
        assert!(
            (hits[1].1 - expected_doc2).abs() < 1e-4,
            "got {}",
            hits[1].1
        );
        assert!(
            (expected_doc2 - 0.231_049).abs() < 1e-3,
            "hand-computed sanity check: {expected_doc2}"
        );

        // And it must measurably differ from the hardcoded-default path.
        let expected_default_doc0 = expected_unnormed_bm25(2, 4, 2.0);
        assert!((hits[0].1 - expected_default_doc0).abs() > 1e-3);

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_scored_with_similarity_top_n_keeps_only_the_best_hit() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let field = "body";
        let term = b"cat";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored_with_similarity(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                lucene_search::similarity::DEFAULT_K1,
                lucene_search::similarity::DEFAULT_B,
                1,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::Ok.code());
        let hits = read_scored_results(scored_handle);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);

        ffi_close_scored_results(scored_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    #[test]
    fn term_query_scored_with_similarity_unknown_segment_handle_is_an_error() {
        let field = "body";
        let term = b"cat";
        let mut scored_handle: u64 = 0;
        let rc = unsafe {
            ffi_search_term_query_scored_with_similarity(
                999_999,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                lucene_search::similarity::DEFAULT_K1,
                lucene_search::similarity::DEFAULT_B,
                10,
                &mut scored_handle as *mut _,
            )
        };
        assert_eq!(rc, FfiStatus::InvalidHandle.code());
    }

    #[test]
    fn term_query_scored_with_similarity_null_out_handle_is_an_error() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let field = "body";
        let term = b"cat";
        let rc = unsafe {
            ffi_search_term_query_scored_with_similarity(
                seg_handle,
                field.as_ptr() as *const c_char,
                field.len(),
                term.as_ptr(),
                term.len(),
                lucene_search::similarity::DEFAULT_K1,
                lucene_search::similarity::DEFAULT_B,
                10,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FfiStatus::NullPointer.code());
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
                crate::legacy_boolean_abi::legacy_search_boolean_query_scored(
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
                crate::legacy_boolean_abi::legacy_search_boolean_query_scored_maxscore(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored_maxscore(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored_maxscore(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored_maxscore(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored_maxscore(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored(
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
            crate::legacy_boolean_abi::legacy_search_boolean_query_scored(
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
        // Since the M2 sweep this registry is an `RwLock` and a query holds
        // only its *read* guard, which `std` never poisons (a shared borrow
        // cannot leave the map half-written). The property under test is
        // unchanged and still worth pinning: whether or not the lock ends up
        // poisoned, a later call must not be wedged by the panic.
        let _ = segments().is_poisoned();

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
    // ------------------------------------------------------------------
    // The occur-tagged clause-array wire format (M2 sweep `c13-ffi-surface`)
    // ------------------------------------------------------------------

    /// Owns the eight parallel arrays one `BooleanQuery` needs, so a test can
    /// describe a clause tree declaratively.
    struct Clauses {
        occurs: Vec<u8>,
        kinds: Vec<u8>,
        fields: Vec<*const c_char>,
        field_lens: Vec<usize>,
        terms: Vec<*const u8>,
        term_lens: Vec<usize>,
        parents: Vec<i32>,
        params: Vec<i32>,
    }

    /// `(occur, kind, field, term, parent, param)`.
    type Spec = (u8, u8, &'static str, &'static [u8], i32, i32);

    impl Clauses {
        fn new(specs: &[Spec]) -> Self {
            let mut c = Clauses {
                occurs: Vec::new(),
                kinds: Vec::new(),
                fields: Vec::new(),
                field_lens: Vec::new(),
                terms: Vec::new(),
                term_lens: Vec::new(),
                parents: Vec::new(),
                params: Vec::new(),
            };
            for (occur, kind, field, term, parent, param) in specs {
                c.occurs.push(*occur);
                c.kinds.push(*kind);
                c.fields.push(field.as_ptr() as *const c_char);
                c.field_lens.push(field.len());
                c.terms.push(term.as_ptr());
                c.term_lens.push(term.len());
                c.parents.push(*parent);
                c.params.push(*param);
            }
            c
        }

        fn search(&self, seg: u64, msm: i32) -> (i32, u64) {
            let mut out: u64 = 0;
            let rc = unsafe {
                ffi_search_boolean_query(
                    seg,
                    self.occurs.as_ptr(),
                    self.kinds.as_ptr(),
                    self.fields.as_ptr(),
                    self.field_lens.as_ptr(),
                    self.terms.as_ptr(),
                    self.term_lens.as_ptr(),
                    self.parents.as_ptr(),
                    self.params.as_ptr(),
                    self.occurs.len(),
                    msm,
                    &mut out as *mut _,
                )
            };
            (rc, out)
        }
    }

    fn term(occur: u8, field: &'static str, t: &'static [u8]) -> Spec {
        (occur, CLAUSE_KIND_TERM, field, t, -1, 0)
    }

    /// A `FILTER` clause narrows exactly like a `MUST` one -- Java's
    /// `Occur.FILTER` is "like MUST except that these clauses do not
    /// participate in scoring". The unscored path cannot see the score
    /// difference, so this asserts the *matched set* is identical, which is
    /// the half `Occur.FILTER` must not change.
    #[test]
    fn a_filter_clause_matches_exactly_what_the_same_must_clause_matches() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);

        let as_must = Clauses::new(&[
            term(OCCUR_MUST, "body", b"cat"),
            term(OCCUR_MUST, "body", b"dog"),
        ]);
        let as_filter = Clauses::new(&[
            term(OCCUR_MUST, "body", b"cat"),
            term(OCCUR_FILTER, "body", b"dog"),
        ]);
        let (rc, h1) = as_must.search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let (rc, h2) = as_filter.search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let (must_docs, filter_docs) = (read_results(h1), read_results(h2));
        assert_eq!(must_docs, filter_docs);
        assert!(
            !must_docs.is_empty(),
            "the fixture must actually match something for this to mean anything"
        );
        ffi_close_results(h1);
        ffi_close_results(h2);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// A filter-only query still *matches* (Java: `BooleanQuery.rewrite`'s
    /// pure-negative test is `clauses.size() == MUST_NOT count`, which a
    /// filter clause fails), it is not a "no positive clauses" empty result.
    #[test]
    fn a_filter_only_query_matches_rather_than_matching_nothing() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let filter_only = Clauses::new(&[term(OCCUR_FILTER, "body", b"cat")]);
        let must_only = Clauses::new(&[term(OCCUR_MUST, "body", b"cat")]);
        let (rc, h1) = filter_only.search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let (rc, h2) = must_only.search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let docs = read_results(h1);
        assert!(!docs.is_empty());
        assert_eq!(docs, read_results(h2));
        ffi_close_results(h1);
        ffi_close_results(h2);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// The scored path is where `FILTER` is observably different from `MUST`:
    /// the filter clause contributes **zero** to the score.
    #[test]
    fn a_filter_clause_contributes_no_score_where_the_same_must_clause_does() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let scored = |c: &Clauses| -> Vec<(i32, f32)> {
            let mut out: u64 = 0;
            let rc = unsafe {
                ffi_search_boolean_query_scored(
                    seg_handle,
                    c.occurs.as_ptr(),
                    c.kinds.as_ptr(),
                    c.fields.as_ptr(),
                    c.field_lens.as_ptr(),
                    c.terms.as_ptr(),
                    c.term_lens.as_ptr(),
                    c.parents.as_ptr(),
                    c.params.as_ptr(),
                    c.occurs.len(),
                    0,
                    10,
                    &mut out as *mut _,
                )
            };
            assert_eq!(rc, FfiStatus::Ok.code());
            let hits = read_scored_results(out);
            crate::results_scored::ffi_close_scored_results(out);
            hits
        };
        let must = scored(&Clauses::new(&[
            term(OCCUR_MUST, "body", b"cat"),
            term(OCCUR_MUST, "body", b"dog"),
        ]));
        let filter = scored(&Clauses::new(&[
            term(OCCUR_MUST, "body", b"cat"),
            term(OCCUR_FILTER, "body", b"dog"),
        ]));
        assert_eq!(
            must.iter().map(|h| h.0).collect::<Vec<_>>(),
            filter.iter().map(|h| h.0).collect::<Vec<_>>(),
            "same documents"
        );
        assert!(
            filter.iter().zip(&must).all(|(f, m)| f.1 < m.1),
            "every filtered hit must score strictly lower: {filter:?} vs {must:?}"
        );
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// A nested `BOOLEAN` clause: `+body:cat +(body:dog body:bird)`.
    #[test]
    fn a_nested_boolean_clause_is_evaluated_as_its_own_subquery() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        // 0: MUST TERM body:cat
        // 1: MUST BOOLEAN (parent -1)
        // 2: SHOULD TERM body:dog  (parent 1)
        // 3: SHOULD TERM body:bird (parent 1)
        let nested = Clauses::new(&[
            term(OCCUR_MUST, "body", b"cat"),
            (OCCUR_MUST, CLAUSE_KIND_BOOLEAN, "", b"", -1, 0),
            (OCCUR_SHOULD, CLAUSE_KIND_TERM, "body", b"dog", 1, 0),
            (OCCUR_SHOULD, CLAUSE_KIND_TERM, "body", b"bird", 1, 0),
        ]);
        let (rc, nested_handle) = nested.search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let nested_docs = read_results(nested_handle);

        // The flat equivalent of the same logic: cat AND (dog OR bird) is not
        // expressible without nesting, so compare against the union of
        // `cat AND dog` and `cat AND bird`.
        let (rc, a) = Clauses::new(&[
            term(OCCUR_MUST, "body", b"cat"),
            term(OCCUR_MUST, "body", b"dog"),
        ])
        .search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let (rc, b) = Clauses::new(&[
            term(OCCUR_MUST, "body", b"cat"),
            term(OCCUR_MUST, "body", b"bird"),
        ])
        .search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let mut expected: Vec<i32> = read_results(a);
        expected.extend(read_results(b));
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(nested_docs, expected);
        assert!(!expected.is_empty());
        ffi_close_results(nested_handle);
        ffi_close_results(a);
        ffi_close_results(b);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// `minimumNumberShouldMatch`, for the root query and for a nested one:
    /// both had no wire representation at all before this batch.
    #[test]
    fn minimum_should_match_narrows_the_result_at_the_root_and_when_nested() {
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let three_shoulds = Clauses::new(&[
            term(OCCUR_SHOULD, "body", b"cat"),
            term(OCCUR_SHOULD, "body", b"dog"),
            term(OCCUR_SHOULD, "body", b"bird"),
        ]);
        let (rc, any) = three_shoulds.search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        let (rc, two) = three_shoulds.search(seg_handle, 2);
        assert_eq!(rc, FfiStatus::Ok.code());
        let (any_docs, two_docs) = (read_results(any), read_results(two));
        assert!(two_docs.len() < any_docs.len(), "mSM=2 must narrow");
        assert!(two_docs.iter().all(|d| any_docs.contains(d)));

        // The same three clauses nested under one BOOLEAN clause with its own
        // mSM must produce the same set as the root-level mSM above.
        let nested = Clauses::new(&[
            (OCCUR_MUST, CLAUSE_KIND_BOOLEAN, "", b"", -1, 2),
            (OCCUR_SHOULD, CLAUSE_KIND_TERM, "body", b"cat", 0, 0),
            (OCCUR_SHOULD, CLAUSE_KIND_TERM, "body", b"dog", 0, 0),
            (OCCUR_SHOULD, CLAUSE_KIND_TERM, "body", b"bird", 0, 0),
        ]);
        let (rc, nested_handle) = nested.search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(read_results(nested_handle), two_docs);
        ffi_close_results(any);
        ffi_close_results(two);
        ffi_close_results(nested_handle);
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// Clause order inside each bucket is the caller's, not the reverse the
    /// bottom-up build pass naturally produces.
    #[test]
    fn clause_order_within_a_bucket_is_preserved() {
        let query = unsafe {
            let c = Clauses::new(&[
                term(OCCUR_SHOULD, "f", b"a"),
                term(OCCUR_SHOULD, "f", b"b"),
                term(OCCUR_SHOULD, "f", b"c"),
                (OCCUR_MUST, CLAUSE_KIND_BOOLEAN, "", b"", -1, 0),
                (OCCUR_MUST, CLAUSE_KIND_TERM, "f", b"x", 3, 0),
                (OCCUR_MUST, CLAUSE_KIND_TERM, "f", b"y", 3, 0),
            ]);
            read_boolean_query(
                c.occurs.as_ptr(),
                c.kinds.as_ptr(),
                c.fields.as_ptr(),
                c.field_lens.as_ptr(),
                c.terms.as_ptr(),
                c.term_lens.as_ptr(),
                c.parents.as_ptr(),
                c.params.as_ptr(),
                6,
                0,
            )
            .unwrap()
        };
        let shoulds: Vec<&[u8]> = query
            .should
            .iter()
            .map(|c| match c {
                Clause::Term(t) => t.term.as_slice(),
                _ => panic!("expected a term clause"),
            })
            .collect();
        assert_eq!(shoulds, vec![b"a".as_slice(), b"b", b"c"]);
        let Clause::Boolean(nested) = &query.must[0] else {
            panic!("expected the nested boolean clause");
        };
        let musts: Vec<&[u8]> = nested
            .must
            .iter()
            .map(|c| match c {
                Clause::Term(t) => t.term.as_slice(),
                _ => panic!("expected a term clause"),
            })
            .collect();
        assert_eq!(musts, vec![b"x".as_slice(), b"y"]);
    }

    /// Null `clause_parents`/`clause_params` mean "flat, all defaults" -- the
    /// convenience the common case relies on.
    #[test]
    fn null_parents_and_params_mean_a_flat_default_query() {
        let field = "body";
        let term_bytes = b"cat";
        let occurs = [OCCUR_MUST];
        let kinds = [CLAUSE_KIND_TERM];
        let fields = [field.as_ptr() as *const c_char];
        let field_lens = [field.len()];
        let terms = [term_bytes.as_ptr()];
        let term_lens = [term_bytes.len()];
        let query = unsafe {
            read_boolean_query(
                occurs.as_ptr(),
                kinds.as_ptr(),
                fields.as_ptr(),
                field_lens.as_ptr(),
                terms.as_ptr(),
                term_lens.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                0,
            )
            .unwrap()
        };
        assert_eq!(query.must.len(), 1);
        assert!(query.filter.is_empty() && query.should.is_empty());
        assert_eq!(query.minimum_should_match, 0);
    }

    fn decode_err(specs: &[Spec], msm: i32) -> FfiStatus {
        let c = Clauses::new(specs);
        unsafe {
            read_boolean_query(
                c.occurs.as_ptr(),
                c.kinds.as_ptr(),
                c.fields.as_ptr(),
                c.field_lens.as_ptr(),
                c.terms.as_ptr(),
                c.term_lens.as_ptr(),
                c.parents.as_ptr(),
                c.params.as_ptr(),
                c.occurs.len(),
                msm,
            )
            .expect_err("expected a rejected clause array")
        }
    }

    #[test]
    fn every_malformed_clause_array_is_an_invalid_argument_not_a_panic() {
        // Unknown Occur tag.
        assert_eq!(
            decode_err(&[(4, CLAUSE_KIND_TERM, "f", b"a", -1, 0)], 0),
            FfiStatus::InvalidArgument
        );
        assert!(last_error_message().contains("unknown Occur tag 4"));
        // Unknown clause kind.
        assert_eq!(
            decode_err(&[(OCCUR_MUST, 7, "f", b"a", -1, 0)], 0),
            FfiStatus::InvalidArgument
        );
        assert!(last_error_message().contains("unknown clause kind 7"));
        // A forward parent reference (which is also how a cycle would have to
        // be spelled).
        assert_eq!(
            decode_err(
                &[
                    (OCCUR_MUST, CLAUSE_KIND_TERM, "f", b"a", 1, 0),
                    (OCCUR_MUST, CLAUSE_KIND_BOOLEAN, "", b"", -1, 0),
                ],
                0
            ),
            FfiStatus::InvalidArgument
        );
        assert!(last_error_message().contains("an earlier clause"));
        // A self parent.
        assert_eq!(
            decode_err(&[(OCCUR_MUST, CLAUSE_KIND_TERM, "f", b"a", 0, 0)], 0),
            FfiStatus::InvalidArgument
        );
        // A parent that is a leaf.
        assert_eq!(
            decode_err(
                &[
                    (OCCUR_MUST, CLAUSE_KIND_TERM, "f", b"a", -1, 0),
                    (OCCUR_MUST, CLAUSE_KIND_TERM, "f", b"b", 0, 0),
                ],
                0
            ),
            FfiStatus::InvalidArgument
        );
        assert!(last_error_message().contains("is not a BOOLEAN clause"));
        // A non-zero param on a TERM clause is reserved, not ignored.
        assert_eq!(
            decode_err(&[(OCCUR_MUST, CLAUSE_KIND_TERM, "f", b"a", -1, 3)], 0),
            FfiStatus::InvalidArgument
        );
        assert!(last_error_message().contains("must be 0 for a TERM clause"));
        // A negative nested minimumNumberShouldMatch.
        assert_eq!(
            decode_err(&[(OCCUR_MUST, CLAUSE_KIND_BOOLEAN, "", b"", -1, -1)], 0),
            FfiStatus::InvalidArgument
        );
        // A negative root minimumNumberShouldMatch.
        assert_eq!(
            decode_err(&[term(OCCUR_MUST, "f", b"a")], -1),
            FfiStatus::InvalidArgument
        );
        assert!(last_error_message().contains("minimumNumberShouldMatch -1 is negative"));
    }

    /// A caller-controlled nesting depth is a caller-controlled *stack* depth
    /// once the query is evaluated and dropped, and a stack overflow aborts --
    /// which `catch_unwind` cannot contain. The cap must reject before that.
    #[test]
    fn nesting_deeper_than_the_cap_is_rejected() {
        let mut specs: Vec<Spec> = Vec::new();
        for i in 0..MAX_CLAUSE_DEPTH + 1 {
            specs.push((OCCUR_MUST, CLAUSE_KIND_BOOLEAN, "", b"", i as i32 - 1, 0));
        }
        assert_eq!(decode_err(&specs, 0), FfiStatus::InvalidArgument);
        assert!(last_error_message().contains("nesting depth"));

        // Exactly at the cap is still accepted.
        let mut ok: Vec<Spec> = Vec::new();
        for i in 0..MAX_CLAUSE_DEPTH {
            ok.push((OCCUR_MUST, CLAUSE_KIND_BOOLEAN, "", b"", i as i32 - 1, 0));
        }
        let c = Clauses::new(&ok);
        let query = unsafe {
            read_boolean_query(
                c.occurs.as_ptr(),
                c.kinds.as_ptr(),
                c.fields.as_ptr(),
                c.field_lens.as_ptr(),
                c.terms.as_ptr(),
                c.term_lens.as_ptr(),
                c.parents.as_ptr(),
                c.params.as_ptr(),
                c.occurs.len(),
                0,
            )
        };
        assert!(query.is_ok());
    }

    /// The clause cap is one array length now, so the old
    /// "three lists of 1024 each" hole cannot recur.
    #[test]
    fn the_clause_cap_is_the_whole_array_and_counts_nested_clauses_too() {
        assert_eq!(check_clause_count(MAX_CLAUSE_COUNT), Ok(()));
        assert_eq!(
            check_clause_count(MAX_CLAUSE_COUNT + 1),
            Err(FfiStatus::InvalidArgument)
        );
        assert!(last_error_message().contains("maxClauseCount"));
        // A whole query of 1025 clauses is refused at the entry point, not
        // just by the helper.
        let dir_handle = open_dir();
        let seg_handle = open_segment(dir_handle, false);
        let specs: Vec<Spec> = (0..MAX_CLAUSE_COUNT + 1)
            .map(|_| term(OCCUR_SHOULD, "body", b"cat"))
            .collect();
        let (rc, _) = Clauses::new(&specs).search(seg_handle, 0);
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        ffi_close_segment(seg_handle);
        ffi_close_directory(dir_handle);
    }

    /// `clause_field_names` must find every field in the tree, including ones
    /// only a nested clause mentions -- otherwise a nested clause's norms are
    /// silently missing and its scores fall back to the unnormed constant.
    #[test]
    fn clause_field_names_walks_nested_clauses() {
        let c = Clauses::new(&[
            term(OCCUR_MUST, "outer", b"a"),
            (OCCUR_MUST, CLAUSE_KIND_BOOLEAN, "", b"", -1, 0),
            (OCCUR_SHOULD, CLAUSE_KIND_TERM, "inner", b"b", 1, 0),
            (OCCUR_FILTER, CLAUSE_KIND_TERM, "outer", b"c", 1, 0),
        ]);
        let query = unsafe {
            read_boolean_query(
                c.occurs.as_ptr(),
                c.kinds.as_ptr(),
                c.fields.as_ptr(),
                c.field_lens.as_ptr(),
                c.terms.as_ptr(),
                c.term_lens.as_ptr(),
                c.parents.as_ptr(),
                c.params.as_ptr(),
                4,
                0,
            )
            .unwrap()
        };
        let mut names = clause_field_names(&query);
        names.sort_unstable();
        assert_eq!(names, vec!["inner", "outer"]);
    }

    /// Null `clause_occurs`/`clause_kinds` with a non-zero count is a status
    /// code, never a dereference.
    #[test]
    fn a_null_clause_array_with_a_nonzero_count_is_a_null_pointer_error() {
        let e = unsafe {
            read_boolean_query(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                3,
                0,
            )
        };
        assert_eq!(e, Err(FfiStatus::NullPointer));
        // A TERM clause with null field/term arrays is likewise refused.
        let occurs = [OCCUR_MUST];
        let kinds = [CLAUSE_KIND_TERM];
        let e = unsafe {
            read_boolean_query(
                occurs.as_ptr(),
                kinds.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                0,
            )
        };
        assert_eq!(e, Err(FfiStatus::NullPointer));
        // Zero clauses is a valid, match-nothing query.
        let q = unsafe {
            read_boolean_query(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
            )
            .unwrap()
        };
        assert!(q.must.is_empty() && q.should.is_empty());
    }
}
