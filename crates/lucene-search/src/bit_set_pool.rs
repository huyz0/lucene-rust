//! Cleared bit sets kept per thread for the next query to fill instead of
//! allocating.
//!
//! A one-bit-per-document set over a 1M-document segment is 128 KiB, and a
//! fresh allocation of that size comes back as untouched pages that fault in
//! on first write: about 27 page faults per query for a numeric sort's
//! competitive iterator (`top_field`), about 10% of a constant-score
//! multi-term query's profile (`exec::multi_term`). Lucene allocates the same
//! sets per query, but on a warm heap where a new `long[]` is a memset;
//! reusing a cleared set is what gives Rust the same effect. Sets are keyed by
//! length, because a set fits only a segment of exactly its `maxDoc`, and a
//! thread keeps at most [`SPARE_BIT_SET_LIMIT_BYTES`] of them.

use std::cell::RefCell;

use lucene_util::fixed_bit_set::FixedBitSet;

/// The most a thread keeps, in bytes of bit storage: four 1M-document
/// sets' worth several times over, and nothing near a large segment's
/// set, which is dropped rather than pinned.
pub(crate) const SPARE_BIT_SET_LIMIT_BYTES: usize = 8 << 20;

thread_local! {
    static SPARE: RefCell<Vec<FixedBitSet>> = const { RefCell::new(Vec::new()) };
}

fn bytes(b: &FixedBitSet) -> usize {
    b.len().div_ceil(64) * 8
}

/// A cleared set of exactly `len` bits, if this thread has one spare.
pub(crate) fn take(len: usize) -> Option<FixedBitSet> {
    SPARE.with(|s| {
        let mut s = s.borrow_mut();
        let at = s.iter().position(|b| b.len() == len)?;
        Some(s.swap_remove(at))
    })
}

/// Clears `b` and keeps it for a later [`take`], unless the thread's
/// spares would pass the limit.
pub(crate) fn give(mut b: FixedBitSet) {
    if bytes(&b) > SPARE_BIT_SET_LIMIT_BYTES {
        return;
    }
    SPARE.with(|s| {
        let mut s = s.borrow_mut();
        let held: usize = s.iter().map(bytes).sum();
        if held + bytes(&b) <= SPARE_BIT_SET_LIMIT_BYTES {
            b.clear_all();
            s.push(b);
        }
    });
}

/// How many sets of `len` bits this thread holds (tests).
#[cfg(test)]
pub(crate) fn held(len: usize) -> usize {
    SPARE.with(|s| s.borrow().iter().filter(|b| b.len() == len).count())
}
