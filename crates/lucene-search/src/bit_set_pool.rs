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
//! length, because a set fits only a segment of exactly its `maxDoc`.
//!
//! A thread keeps at most [`SPARE_BIT_SET_LIMIT_BYTES`] of them and at most
//! [`SPARE_BIT_SET_LIMIT_COUNT`] sets, the oldest going first when a new one
//! would pass either. The count bound is what keeps [`take`] and [`give`]
//! constant-time: every cached bit set is handed back here when it is
//! dropped (`exec::cache::CachedSet`), including the sets of point ranges no
//! later query ever takes, and with a byte limit alone a run of queries over
//! small segments piled up hundreds of thousands of tiny sets that every
//! call then scanned -- `m7_fixture`'s point queries went from 1.39x of
//! Lucene to 0.04x that way (`docs/benchmarks/m7-2026-10.md`).

use std::cell::RefCell;
use std::collections::VecDeque;

use lucene_util::fixed_bit_set::FixedBitSet;

/// The most a thread keeps, in bytes of bit storage: four 1M-document
/// sets' worth several times over, and nothing near a large segment's
/// set, which is dropped rather than pinned.
pub(crate) const SPARE_BIT_SET_LIMIT_BYTES: usize = 8 << 20;

/// The most sets a thread keeps, whatever their size: a query holds a few
/// sets per segment at once, so this covers a few queries' worth over a
/// handful of segments while keeping every scan of the pool short.
pub(crate) const SPARE_BIT_SET_LIMIT_COUNT: usize = 32;

/// One thread's spares, oldest first, and their total bytes.
struct Spare {
    sets: VecDeque<FixedBitSet>,
    bytes: usize,
}

thread_local! {
    static SPARE: RefCell<Spare> = const {
        RefCell::new(Spare {
            sets: VecDeque::new(),
            bytes: 0,
        })
    };
}

fn bytes(b: &FixedBitSet) -> usize {
    b.len().div_ceil(64) * 8
}

/// A cleared set of exactly `len` bits, if this thread has one spare.
///
/// Sets are cleared here rather than in [`give`]: most sets given back are
/// never taken (a point range's, a segment no later query revisits), and
/// clearing those would be a memset of up to the whole set per query for
/// nothing -- `geo_points_contains` read 0.78x of the unbounded pool's time
/// that way.
pub(crate) fn take(len: usize) -> Option<FixedBitSet> {
    let mut b = SPARE.with(|s| {
        let mut s = s.borrow_mut();
        // The newest first: the set a query just finished with.
        let at = s.sets.iter().rposition(|b| b.len() == len)?;
        let b = s.sets.remove(at)?;
        s.bytes -= bytes(&b);
        Some(b)
    })?;
    b.clear_all();
    Some(b)
}

/// Keeps `b`, as it is, for a later [`take`] (which clears it), dropping
/// the oldest spares as needed to stay within both limits.
pub(crate) fn give(b: FixedBitSet) {
    let size = bytes(&b);
    if size > SPARE_BIT_SET_LIMIT_BYTES {
        return;
    }
    SPARE.with(|s| {
        let mut s = s.borrow_mut();
        while s.sets.len() >= SPARE_BIT_SET_LIMIT_COUNT
            || s.bytes + size > SPARE_BIT_SET_LIMIT_BYTES
        {
            let Some(old) = s.sets.pop_front() else { break };
            s.bytes -= bytes(&old);
        }
        s.bytes += size;
        s.sets.push_back(b);
    });
}

/// How many sets of `len` bits this thread holds (tests).
#[cfg(test)]
pub(crate) fn held(len: usize) -> usize {
    SPARE.with(|s| s.borrow().sets.iter().filter(|b| b.len() == len).count())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    fn totals() -> (usize, usize) {
        SPARE.with(|s| {
            let s = s.borrow();
            assert_eq!(s.bytes, s.sets.iter().map(bytes).sum::<usize>());
            (s.sets.len(), s.bytes)
        })
    }

    #[test]
    fn many_sets_no_query_takes_back_stay_bounded() {
        // What a run of point-range queries over small segments hands back:
        // sets of lengths nothing takes. The pool keeps the newest few.
        for i in 0..10 * SPARE_BIT_SET_LIMIT_COUNT {
            give(FixedBitSet::new(180 + i % 3));
        }
        let (count, _) = totals();
        assert_eq!(count, SPARE_BIT_SET_LIMIT_COUNT);
        // A set a later query wants is still kept, and is the one it gets.
        let mut wanted = FixedBitSet::new(4_099);
        wanted.set(5);
        give(wanted);
        assert_eq!(totals().0, SPARE_BIT_SET_LIMIT_COUNT);
        let back = take(4_099).expect("the newest set is kept");
        assert_eq!(back.cardinality(), 0, "handed back cleared");
        assert_eq!(totals().0, SPARE_BIT_SET_LIMIT_COUNT - 1);
    }

    #[test]
    fn a_set_that_would_pass_the_byte_limit_evicts_the_oldest() {
        // Just over half the limit: two never fit together.
        let half = SPARE_BIT_SET_LIMIT_BYTES * 8 / 2 + 64;
        give(FixedBitSet::new(64));
        give(FixedBitSet::new(half));
        assert_eq!((held(64), held(half)), (1, 1));
        // The second one fits only once both older sets are gone.
        give(FixedBitSet::new(half));
        assert_eq!((held(64), held(half)), (0, 1));
        assert_eq!(totals(), (1, half / 8));
        assert!(take(half).is_some());
        assert_eq!(totals(), (0, 0));
        assert!(take(half).is_none());
    }
}
