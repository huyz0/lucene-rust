//! **Test-only.** The pre-`c13-ffi-surface` three-bucket boolean-query ABI,
//! re-expressed on top of the occur-tagged clause array every
//! `ffi_search_boolean_query*`/`ffi_explain_boolean_query` entry point now
//! takes (see [`crate::query::read_boolean_query`]).
//!
//! **Why this exists**: the M2 sweep batch `c13-ffi-surface` replaced three
//! separate `must`/`should`/`must_not` four-array clause lists with one
//! `Occur`-tagged, parent-indexed clause array, so that `Occur.FILTER` (and
//! any future `Occur`, and nested `Clause::Boolean`) became expressible
//! without another C-ABI break. That change has to be proved *behaviour-
//! preserving* for everything the old format could already say, and the
//! sharpest available proof is the crate's own pre-existing boolean-query
//! test suite -- dozens of assertions about matched doc sets, scores,
//! MAXSCORE pruning, explanations and error codes, written against the old
//! parameter shape. Running them unchanged, through a translation layer that
//! is itself trivial, is a stronger statement than re-deriving each expected
//! value by hand against the new shape would be.
//!
//! Every shim here calls the **real exported symbol** -- nothing is
//! reimplemented, and the null-pointer/handle/clause-count checks the tests
//! assert on are still the exported function's own. New capabilities the old
//! format could not express (`FILTER`, nesting, `minimumNumberShouldMatch`)
//! are covered by tests written directly against the new signature, not
//! through this bridge.

use std::os::raw::c_char;

/// The eight parallel arrays [`crate::query::read_boolean_query`] reads,
/// materialized from three old-style `(fields, field_lens, terms, term_lens,
/// count)` buckets.
pub(crate) struct ClauseArrays {
    pub occurs: Vec<u8>,
    pub kinds: Vec<u8>,
    pub fields: Vec<*const c_char>,
    pub field_lens: Vec<usize>,
    pub terms: Vec<*const u8>,
    pub term_lens: Vec<usize>,
    pub parents: Vec<i32>,
    pub params: Vec<i32>,
}

impl ClauseArrays {
    /// Concatenates the three buckets in Java's own `Occur` declaration
    /// order, tagging each clause with the bucket it came from. Every clause
    /// is a top-level `TERM` clause (`parent == -1`, `param == 0`), which is
    /// exactly what the old format could express.
    ///
    /// Returns `None` when a bucket has a non-zero count but a null array --
    /// the case the old tests assert returns [`crate::error::FfiStatus::NullPointer`].
    /// The caller then forwards null arrays to the real entry point so its
    /// own null check produces that status, rather than this bridge
    /// dereferencing null.
    ///
    /// # Safety
    /// Each non-null `(array, count)` pair must be valid for `count` reads,
    /// exactly as the old exported functions required.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn from_buckets(
        buckets: [(
            u8,
            *const *const c_char,
            *const usize,
            *const *const u8,
            *const usize,
            usize,
        ); 3],
    ) -> Option<Self> {
        let mut out = ClauseArrays {
            occurs: Vec::new(),
            kinds: Vec::new(),
            fields: Vec::new(),
            field_lens: Vec::new(),
            terms: Vec::new(),
            term_lens: Vec::new(),
            parents: Vec::new(),
            params: Vec::new(),
        };
        for (occur, fields, field_lens, terms, term_lens, count) in buckets {
            if count == 0 {
                continue;
            }
            if fields.is_null() || field_lens.is_null() || terms.is_null() || term_lens.is_null() {
                return None;
            }
            for i in 0..count {
                // SAFETY: this function's own contract: each array is valid
                // for `count` elements.
                unsafe {
                    out.fields.push(*fields.add(i));
                    out.field_lens.push(*field_lens.add(i));
                    out.terms.push(*terms.add(i));
                    out.term_lens.push(*term_lens.add(i));
                }
                out.occurs.push(occur);
                out.kinds.push(crate::query::CLAUSE_KIND_TERM);
                out.parents.push(-1);
                out.params.push(0);
            }
        }
        Some(out)
    }

    pub(crate) fn len(&self) -> usize {
        self.occurs.len()
    }
}

/// Generates one shim per exported boolean entry point: same old parameter
/// order, same trailing parameters, forwarding to the real symbol.
macro_rules! legacy_shim {
    ($name:ident => $target:path $(, $tail:ident : $tail_ty:ty)*) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn $name(
            handle: u64,
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
            $($tail: $tail_ty,)*
            out_handle: *mut u64,
        ) -> i32 {
            // SAFETY: forwarded from this shim's caller, which is a test that
            // owns the arrays it passes.
            let arrays = unsafe {
                ClauseArrays::from_buckets([
                    (
                        crate::query::OCCUR_MUST,
                        must_fields,
                        must_field_lens,
                        must_terms,
                        must_term_lens,
                        must_count,
                    ),
                    (
                        crate::query::OCCUR_SHOULD,
                        should_fields,
                        should_field_lens,
                        should_terms,
                        should_term_lens,
                        should_count,
                    ),
                    (
                        crate::query::OCCUR_MUST_NOT,
                        must_not_fields,
                        must_not_field_lens,
                        must_not_terms,
                        must_not_term_lens,
                        must_not_count,
                    ),
                ])
            };
            match arrays {
                Some(c) => {
                    let n = c.len();
                    // SAFETY: every array is a live local `Vec` of `n`
                    // elements; `out_handle` is the caller's.
                    unsafe {
                        $target(
                            handle,
                            c.occurs.as_ptr(),
                            c.kinds.as_ptr(),
                            c.fields.as_ptr(),
                            c.field_lens.as_ptr(),
                            c.terms.as_ptr(),
                            c.term_lens.as_ptr(),
                            c.parents.as_ptr(),
                            c.params.as_ptr(),
                            n,
                            0,
                            $($tail,)*
                            out_handle,
                        )
                    }
                }
                // A null clause array with a non-zero count: hand the nulls
                // straight through so the real function's own null check is
                // what produces the status the test asserts.
                None => {
                    let total = must_count
                        .saturating_add(should_count)
                        .saturating_add(must_not_count);
                    // SAFETY: every clause pointer is null and `total` is only
                    // read after the null check rejects it.
                    unsafe {
                        $target(
                            handle,
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            total,
                            0,
                            $($tail,)*
                            out_handle,
                        )
                    }
                }
            }
        }
    };
}

legacy_shim!(legacy_search_boolean_query => crate::query::ffi_search_boolean_query);
legacy_shim!(legacy_search_boolean_query_scored => crate::query::ffi_search_boolean_query_scored, top_n: usize);
legacy_shim!(legacy_search_boolean_query_scored_maxscore => crate::query::ffi_search_boolean_query_scored_maxscore, top_n: usize);
legacy_shim!(legacy_explain_boolean_query => crate::explain::ffi_explain_boolean_query, doc: i32);
legacy_shim!(legacy_search_boolean_query_multi_segment => crate::directory_reader::ffi_search_boolean_query_multi_segment, top_n: usize);
legacy_shim!(legacy_search_boolean_query_multi_segment_concurrent => crate::directory_reader::ffi_search_boolean_query_multi_segment_concurrent, top_n: usize);
legacy_shim!(legacy_search_boolean_query_multi_segment_maxscore => crate::directory_reader::ffi_search_boolean_query_multi_segment_maxscore, top_n: usize);
legacy_shim!(legacy_search_boolean_query_multi_segment_maxscore_concurrent => crate::directory_reader::ffi_search_boolean_query_multi_segment_maxscore_concurrent, top_n: usize);
