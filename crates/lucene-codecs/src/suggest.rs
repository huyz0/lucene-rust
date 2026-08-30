//! Port of `org.apache.lucene.search.suggest.fst.WFSTCompletionLookup`
//! (simplified from `AnalyzingSuggester`), restricted to the FST *shape* and
//! its weighted-completion-by-prefix lookup.
//!
//! Real `WFSTCompletionLookup` builds a single FST mapping surface-form term
//! bytes to a "weight" (an unsigned popularity/ranking score), using
//! `PositiveIntOutputs` as the output type -- this crate already has that
//! primitive (`crate::fst::PositiveIntOutputs`) plus the underlying
//! `FSTCompiler`/`Fst` machinery (`crate::fst::build_fst_typed`,
//! `crate::fst::Fst`). What's added *here* is the suggester-specific layer on
//! top: the weight-inversion trick during construction (so that the FST's
//! natural minimal-output-along-shared-prefix property surfaces the
//! highest-weight completions first when walking outputs greedily) and a
//! top-N weighted-completion lookup given a prefix.
//!
//! ## Weight inversion: exact arithmetic and why
//!
//! Real Lucene's `WFSTCompletionLookup` stores, for each term, a "cost"
//! `encodeWeight(weight) = Integer.MAX_VALUE - (int) weight` (see
//! `WFSTCompletionLookup.java`'s `encodeWeight`/`decodeWeight`, verified by
//! reading that file directly at
//! `lucene/suggest/src/java/org/apache/lucene/search/suggest/fst/WFSTCompletionLookup.java`
//! rather than reasoning it out from scratch) as the `PositiveIntOutputs`
//! value on the FST, then recovers the weight the same way:
//! `decodeWeight(cost) = Integer.MAX_VALUE - cost`. Since an FST's own
//! machinery (`Util.shortestPaths`, a priority search) naturally surfaces
//! *smallest*-output paths first, inverting the weight this way means the
//! smallest cost -- and therefore the first path found -- corresponds to the
//! *largest* weight, exactly the "highest popularity first" behavior a
//! suggester wants.
//!
//! This port's suggester weight type is `u32` (not Java's `int`/`Integer`,
//! since there is no reason to restrict this Rust API to non-negative
//! signed 31-bit values when `u32` covers the same "unsigned popularity
//! score" concept more directly), so the corresponding constant is `u32::MAX`
//! rather than `Integer.MAX_VALUE`: `cost = u32::MAX - weight` and
//! `weight = u32::MAX - cost`. This is exactly Java's identity, just widened
//! to match this port's weight type -- both formulas are involutions
//! (`decode(encode(w)) == w` for all representable `w`) that reverse the
//! natural ordering (`w1 < w2  <=>  encode(w1) > encode(w2)`), which is the
//! only property the inversion trick actually depends on. `PositiveIntOutputs`
//! stores its value as `i64`, so `u32::MAX - weight` (which is always in
//! `0..=u32::MAX`) never overflows or needs to be negative.
//!
//! ## Scope of this module (explicitly not a full port)
//!
//! - **No `AnalyzingSuggester`-style analysis.** Real Lucene's suggester
//!   stack normally tokenizes/analyzes surface forms through a configurable
//!   `Analyzer` before building the FST (fuzzy matching, multiple surface
//!   forms per weight, deduplication across analyzed forms, etc. --
//!   `AnalyzingSuggester`, not `WFSTCompletionLookup` itself). This module
//!   takes raw `(term_bytes, weight)` pairs, matching `WFSTCompletionLookup`'s
//!   own (simpler) contract, not `AnalyzingSuggester`'s.
//! - **No fuzzy/edit-distance suggestion.** Only exact-prefix continuation is
//!   supported (`top_n_completions`), not `FuzzySuggester`'s Levenshtein-
//!   automaton-based matching.
//! - **No on-disk suggester index format/persistence.** `WFSTCompletionLookup`
//!   supports `store`/`load` over a `DataOutput`/`DataInput` (in addition to
//!   its own `count` field). This module doesn't add a dedicated persistence
//!   format, but -- worth noting explicitly, since it's not free-standing
//!   scope creep -- persistence of the FST itself already falls out of this
//!   crate's existing FST byte format for free: `crate::fst::build_fst_typed`
//!   produces a plain `Fst<'static>` that Java's own `FST.save`/`FST.read`
//!   wire format already round-trips through this crate's `Fst::read`/
//!   `Fst::read_borrowed` (see `fst.rs`'s module doc). A caller that also
//!   wants to persist the suggester's `count` field alongside the FST body
//!   (as `WFSTCompletionLookup.store`/`load` do) can trivially do so with a
//!   `writeVLong`/`readVLong`-equivalent wrapper of their own; this module
//!   doesn't add that wrapper since it isn't part of the FST-shape task this
//!   module exists to cover.
//! - **`top_n_completions` enumerates the prefix's matching completions, then
//!   selects top-N via a bounded (size-`n`) min-heap -- it does not
//!   reproduce real Lucene's `Util.shortestPaths`/`TopNSearcher`, a genuine
//!   priority-queue-based FST walk that can short-circuit *within* the
//!   remaining-suffix search space itself (partially expanding only the most
//!   promising nodes) without ever materializing every matching completion.**
//!   This is a deliberate, disclosed scope reduction, not a silent
//!   under-delivery: `top_n_completions` *does* avoid touching the rest of
//!   the FST outside the prefix's subtree (it seeks directly to the prefix
//!   via `FstEnum::seek_ceil` and then walks only entries in ascending key
//!   order that still share the prefix, stopping the instant one doesn't),
//!   and it keeps only `n` completions in memory at any time (a bounded
//!   min-heap, not "collect everything then sort"). What it does *not* do is
//!   avoid *decoding* every matching completion's weight -- for a prefix with
//!   many more matches than `n`, a true priority search could skip whole
//!   subtrees that provably can't beat the current worst kept candidate.
//!   Given this port's context (small in-memory suggestion dictionaries, not
//!   billions of terms), that full priority-queue machinery isn't worth its
//!   complexity yet -- see `docs/parity.md`'s row for this module.

use crate::fst::{self, BuildError, Fst, Outputs, PositiveIntOutputs};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Upper bound on the heap capacity [`top_n_completions`] reserves before it
/// has seen a single matching term. Past this the heap grows on demand, which
/// costs a handful of reallocations for a genuinely huge `n` and removes an
/// unbounded caller-driven allocation.
const HEAP_RESERVE_CAP: usize = 1024;

/// One weighted completion returned by [`top_n_completions`]: the bytes that
/// continue the queried prefix (i.e. `prefix + suffix` is the full matched
/// term) and its original (un-inverted) weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub suffix: Vec<u8>,
    pub weight: u32,
}

/// `WFSTCompletionLookup.encodeWeight`, widened to `u32`: `weight -> cost`.
/// See this module's doc comment for the exact arithmetic and why it's
/// correct.
// ARITH: `weight` is a `u32`, so `u32::MAX - weight` is in `0..=u32::MAX`.
#[allow(clippy::arithmetic_side_effects)]
fn encode_weight(weight: u32) -> i64 {
    (u32::MAX - weight) as i64
}

/// `WFSTCompletionLookup.decodeWeight`, widened to `u32`: `cost -> weight`.
///
/// `cost` is an FST output, and an FST can be loaded from bytes on disk, so
/// `cost` is *not* bounded to `0..=u32::MAX` however this module's own
/// `encode_weight` behaves -- a corrupt or foreign FST can hold any `i64` the
/// output vlong decodes to. Java's `decodeWeight` is
/// `(int) (Integer.MAX_VALUE - encoded)`, a cast that simply truncates
/// whatever it is handed; this is the same thing one width up, so a garbage
/// cost yields a garbage weight rather than an overflow panic (which is what
/// the `debug_assert` this replaced turned a corrupt suggester FST into).
fn decode_weight(cost: i64) -> u32 {
    (u32::MAX as u64).wrapping_sub(cost as u64) as u32
}

/// Builds a suggester FST from `(term, weight)` pairs (`WFSTCompletionLookup
/// .build`'s FST-construction step, minus the `InputIterator`/temp-directory
/// external-sort machinery real Lucene uses to accept an arbitrarily large,
/// unsorted stream -- this module's caller is expected to already hold its
/// full term list in memory, matching this port's small-dictionary scope).
///
/// `entries` need not be pre-sorted or de-duplicated. Real
/// `WFSTCompletionLookup.build` reads through a `WFSTInputIterator`, a
/// `SortedInputIterator` whose comparator is documented as *"Sortes by
/// BytesRef (ascending) then cost (ascending)"* -- and cost is
/// `Integer.MAX_VALUE - weight`, so ascending cost means **descending
/// weight**. Its dedup loop then skips every entry equal to the previous
/// one, which is why the code's comment reads *"for duplicate suggestions,
/// the best weight is actually added"*: the survivor is the **highest**
/// weight, not the first one the caller happened to pass.
///
/// This function sorts by `(term asc, weight desc)` and keeps the first of
/// each run for exactly that reason. It previously sorted by term alone and
/// relied on the sort's stability, which kept whichever weight came first in
/// the caller's input -- a silently different suggester for any dictionary
/// with a repeated surface form.
pub fn build_suggester_fst(entries: &[(Vec<u8>, u32)]) -> Result<Fst<'static>, BuildError> {
    let mut sorted: Vec<(Vec<u8>, u32)> = entries.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    sorted.dedup_by(|later, earlier| later.0 == earlier.0);

    let costed: Vec<(Vec<u8>, i64)> = sorted
        .into_iter()
        .map(|(term, weight)| (term, encode_weight(weight)))
        .collect();

    fst::build_fst_typed::<PositiveIntOutputs>(&costed)
}

/// Walks every accepted term of `fst` that starts with `prefix`, returning
/// the `n` highest-weight ones (highest first; ties broken by ascending
/// suffix byte order, which is `Util.TieBreakByInputComparator`'s own
/// tie-break: *"Compares first by the provided comparator, and then tie
/// breaks by path.input"*), each as the suffix continuing `prefix` plus its
/// (un-inverted) weight.
///
/// `exact_first` is `WFSTCompletionLookup`'s constructor flag of the same
/// name, whose default is **`true`** (`WFSTCompletionLookup(Directory,
/// String)` delegates to `(dir, prefix, true)`): when the queried prefix is
/// itself an indexed term, it is returned as the first result *regardless of
/// its weight*, and the remaining `n - 1` slots are filled by weight. Java
/// implements this by emitting the exact hit up front and then calling
/// `Util.shortestPaths(..., allowEmptyString = !exactFirst)` so the same term
/// cannot come back twice. Its javadoc notes the trade-off: *"This has no
/// performance impact, but could result in low-quality suggestions."*
///
/// Returns fewer than `n` completions if fewer than `n` terms share the
/// prefix (including zero). See this module's doc comment ("Scope of this
/// module") for exactly what "top-N" search strategy this does and does not
/// implement.
pub fn top_n_completions(
    fst: &Fst,
    prefix: &[u8],
    n: usize,
    exact_first: bool,
) -> fst::Result<Vec<Completion>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // `exactFirst` reserves one of the `n` slots for the exact hit before the
    // weighted search runs (Java decrements `num` after adding it), so the
    // heap below is sized for what is left.
    let exact_weight = if exact_first {
        fst.get_typed::<PositiveIntOutputs>(prefix)?
            .map(decode_weight)
    } else {
        None
    };
    let n = match exact_weight {
        Some(_) if n == 1 => {
            return Ok(vec![Completion {
                suffix: Vec::new(),
                weight: exact_weight.expect("just matched Some"),
            }]);
        }
        // ARITH: `n == 0` returned above and `n == 1` is the arm just
        // before this one, so `n >= 2` here.
        #[allow(clippy::arithmetic_side_effects)]
        Some(_) => n - 1,
        None => n,
    };

    let mut iter = fst.iter()?;
    let first = iter.seek_ceil(prefix)?;

    // Bounded min-heap: at most `n` entries kept at any time, ordered so the
    // *worst* (lowest-weight) kept candidate is always the one popped when a
    // better candidate needs room. `Reverse` turns `BinaryHeap`'s default
    // max-heap into the min-heap this eviction policy needs.
    // `n` is the caller's "top N" and is not bounded by anything in the
    // index, so reserving `n + 1` slots up front turns a large `n` into an
    // allocation failure -- an abort, which `catch_unwind` cannot intercept.
    // The heap's real occupancy is `min(n, terms sharing the prefix) + 1`, so
    // pre-reserve only up to `HEAP_RESERVE_CAP` and let it grow from there.
    let mut heap: BinaryHeap<Reverse<HeapItem>> =
        BinaryHeap::with_capacity(n.min(HEAP_RESERVE_CAP).saturating_add(1));

    let push = |key: Vec<u8>,
                output: Vec<u8>,
                heap: &mut BinaryHeap<Reverse<HeapItem>>|
     -> std::result::Result<(), fst::Error> {
        let suffix = key[prefix.len()..].to_vec();
        // `allowEmptyString = !exactFirst`: with the exact hit already
        // emitted, `Util.shortestPaths` is told not to return the empty
        // continuation again.
        if exact_weight.is_some() && suffix.is_empty() {
            return Ok(());
        }
        // The cost payload comes off the FST body, so a corrupt one is a
        // decode error, not a panic -- see `fst::Outputs::decode`.
        let cost = PositiveIntOutputs::decode(&output)?;
        let weight = decode_weight(cost);
        heap.push(Reverse(HeapItem { weight, suffix }));
        if heap.len() > n {
            heap.pop();
        }
        Ok(())
    };

    match first {
        Some((key, output)) if key.starts_with(prefix) => push(key, output, &mut heap)?,
        _ => {
            return Ok(exact_weight
                .map(|weight| Completion {
                    suffix: Vec::new(),
                    weight,
                })
                .into_iter()
                .collect())
        }
    }

    for item in iter {
        let (key, output) = item?;
        if !key.starts_with(prefix) {
            break;
        }
        push(key, output, &mut heap)?;
    }

    let mut results: Vec<HeapItem> = heap.into_iter().map(|Reverse(item)| item).collect();
    results.sort_by(|a, b| {
        b.weight
            .cmp(&a.weight)
            .then_with(|| a.suffix.cmp(&b.suffix))
    });

    let exact = exact_weight.map(|weight| Completion {
        suffix: Vec::new(),
        weight,
    });
    Ok(exact
        .into_iter()
        .chain(results.into_iter().map(|item| Completion {
            suffix: item.suffix,
            weight: item.weight,
        }))
        .collect())
}

/// Ordering key for `top_n_completions`'s bounded heap: primarily by weight,
/// then by suffix (ascending) as a deterministic tie-break. Deriving `Ord`
/// gives exactly that (lexicographic over the struct's fields in
/// declaration order).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HeapItem {
    weight: u32,
    suffix: Vec<u8>,
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    /// An FST's outputs come off disk, so a suggester FST can hold a cost far
    /// outside the `0..=u32::MAX` range `encode_weight` produces.
    /// `u32::MAX as i64 - cost` overflowed on such a cost -- a panic in a
    /// debug build, straight through the FFI. Java's `decodeWeight` is a
    /// plain `(int)` cast and cannot fail; this must not either.
    #[test]
    fn out_of_range_cost_decodes_to_garbage_rather_than_panicking() {
        // `u32::MAX - i64::MAX` truncated to 32 bits: 0xFFFFFFFF - 0xFFFFFFFF
        // (the low half of i64::MAX) == 0.
        assert_eq!(decode_weight(i64::MAX), 0);
        // `i64::MIN`'s low 32 bits are 0, so this is `u32::MAX` itself.
        assert_eq!(decode_weight(i64::MIN), u32::MAX);
        // And the in-range identity still holds.
        assert_eq!(decode_weight(0), u32::MAX);
        assert_eq!(decode_weight(u32::MAX as i64), 0);

        // Reached the way a corrupt FST would reach it: through the public
        // entry point, on an FST whose output does not fit `u32`.
        let fst = crate::fst::build_fst_typed::<PositiveIntOutputs>(&[(b"ap".to_vec(), i64::MAX)])
            .unwrap();
        let got = top_n_completions(&fst, b"a", 5, true).unwrap();
        assert_eq!(got.len(), 1);
    }

    /// `n` is the caller's "top N" and is not bounded by the index. Reserving
    /// `n + 1` heap slots up front made a large `n` an allocation failure --
    /// an abort, not a catchable panic.
    #[test]
    fn absurd_n_does_not_reserve_a_slot_per_requested_result() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();
        let got = top_n_completions(&fst, b"ban", usize::MAX, true).unwrap();
        assert_eq!(got.len(), 4);
    }

    fn sample_entries() -> Vec<(Vec<u8>, u32)> {
        vec![
            (b"apple".to_vec(), 10),
            (b"application".to_vec(), 50),
            (b"apply".to_vec(), 30),
            (b"appetite".to_vec(), 5),
            (b"banana".to_vec(), 100),
            (b"band".to_vec(), 20),
            (b"bandana".to_vec(), 20), // tie with "band" on weight
            (b"bandit".to_vec(), 60),
        ]
    }

    #[test]
    fn encode_decode_weight_is_involution() {
        for w in [0u32, 1, 2, 1000, u32::MAX - 1, u32::MAX] {
            assert_eq!(decode_weight(encode_weight(w)), w);
        }
    }

    #[test]
    fn encode_weight_reverses_order() {
        // Higher weight -> lower cost, so minimal-output-first traversal
        // surfaces it first.
        assert!(encode_weight(100) < encode_weight(50));
        assert!(encode_weight(0) > encode_weight(u32::MAX));
    }

    #[test]
    fn build_suggester_fst_round_trips_exact_lookup() {
        let entries = sample_entries();
        let fst = build_suggester_fst(&entries).unwrap();
        for (term, weight) in &entries {
            let cost = fst.get_typed::<PositiveIntOutputs>(term).unwrap().unwrap();
            assert_eq!(decode_weight(cost), *weight);
        }
        assert!(fst
            .get_typed::<PositiveIntOutputs>(b"missing")
            .unwrap()
            .is_none());
    }

    /// `SortedInputIterator`'s comparator sorts by term then by *cost*
    /// ascending, and cost is `Integer.MAX_VALUE - weight`, so the entry that
    /// survives `WFSTCompletionLookup.build`'s dedup is the highest-weight
    /// one regardless of input order. Both orderings are checked, since a
    /// stable sort on the term alone passes one of them by accident.
    #[test]
    fn build_suggester_fst_dedups_keeping_the_highest_weight() {
        for entries in [
            vec![
                (b"dup".to_vec(), 5),
                (b"dup".to_vec(), 999),
                (b"other".to_vec(), 1),
            ],
            vec![
                (b"dup".to_vec(), 999),
                (b"dup".to_vec(), 5),
                (b"other".to_vec(), 1),
            ],
        ] {
            let fst = build_suggester_fst(&entries).unwrap();
            let cost = fst
                .get_typed::<PositiveIntOutputs>(b"dup")
                .unwrap()
                .unwrap();
            assert_eq!(decode_weight(cost), 999);
        }
    }

    #[test]
    fn build_suggester_fst_accepts_unsorted_input() {
        let mut entries = sample_entries();
        entries.reverse();
        let fst = build_suggester_fst(&entries).unwrap();
        let cost = fst
            .get_typed::<PositiveIntOutputs>(b"banana")
            .unwrap()
            .unwrap();
        assert_eq!(decode_weight(cost), 100);
    }

    #[test]
    fn top_n_completions_orders_by_weight_descending() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();

        // "app" has 4 completions: apple(10), application(50), apply(30),
        // appetite(5). Top 2 by weight: application(50), apply(30).
        let top = top_n_completions(&fst, b"app", 2, false).unwrap();
        assert_eq!(
            top,
            vec![
                Completion {
                    suffix: b"lication".to_vec(),
                    weight: 50
                },
                Completion {
                    suffix: b"ly".to_vec(),
                    weight: 30
                },
            ]
        );

        // All 4, in full descending order.
        let all = top_n_completions(&fst, b"app", 10, false).unwrap();
        assert_eq!(
            all.iter().map(|c| c.weight).collect::<Vec<_>>(),
            vec![50, 30, 10, 5]
        );
    }

    #[test]
    fn top_n_completions_breaks_ties_by_suffix_ascending() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();

        // "band"(20) and "bandana"(20) tie in weight; "bandit"(60) wins
        // outright. Prefix "band" completions: ""(20), "ana"(20), "it"(60).
        let top = top_n_completions(&fst, b"band", 3, false).unwrap();
        assert_eq!(
            top,
            vec![
                Completion {
                    suffix: b"it".to_vec(),
                    weight: 60
                },
                Completion {
                    suffix: Vec::new(),
                    weight: 20
                },
                Completion {
                    suffix: b"ana".to_vec(),
                    weight: 20
                },
            ]
        );
    }

    #[test]
    fn top_n_completions_prefix_matching_exact_single_term() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();
        // "banana" has no other term sharing it as a proper prefix.
        let top = top_n_completions(&fst, b"banana", 5, false).unwrap();
        assert_eq!(
            top,
            vec![Completion {
                suffix: Vec::new(),
                weight: 100
            }]
        );
    }

    #[test]
    fn top_n_completions_returns_empty_for_unmatched_prefix() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();
        assert_eq!(
            top_n_completions(&fst, b"zzz", 5, false).unwrap(),
            Vec::new()
        );
        // Prefix strictly beyond the last key too.
        assert_eq!(
            top_n_completions(&fst, b"zzzzzzz", 5, false).unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn top_n_completions_n_zero_returns_empty() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();
        assert_eq!(
            top_n_completions(&fst, b"app", 0, false).unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn top_n_completions_n_larger_than_available_returns_all() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();
        let top = top_n_completions(&fst, b"apple", 1000, false).unwrap();
        assert_eq!(
            top,
            vec![Completion {
                suffix: Vec::new(),
                weight: 10
            }]
        );
    }

    #[test]
    fn top_n_completions_empty_prefix_covers_whole_dictionary() {
        let entries = sample_entries();
        let fst = build_suggester_fst(&entries).unwrap();
        let top = top_n_completions(&fst, b"", 100, false).unwrap();
        assert_eq!(top.len(), entries.len());
        assert_eq!(top[0].weight, 100); // "banana"
                                        // Descending order maintained throughout.
        for w in top.windows(2) {
            assert!(w[0].weight >= w[1].weight);
        }
    }

    /// `WFSTCompletionLookup`'s `exactFirst` default (`true`): when the
    /// queried prefix is itself an indexed term it comes back first no matter
    /// how low its weight is, and never a second time from the weighted
    /// search (`Util.shortestPaths(..., allowEmptyString = !exactFirst)`).
    #[test]
    fn top_n_completions_exact_first_puts_the_exact_hit_ahead_of_heavier_ones() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();

        // "band"(20) is an indexed term, and "bandit"(60) outweighs it.
        let top = top_n_completions(&fst, b"band", 3, true).unwrap();
        assert_eq!(
            top,
            vec![
                Completion {
                    suffix: Vec::new(),
                    weight: 20
                },
                Completion {
                    suffix: b"it".to_vec(),
                    weight: 60
                },
                Completion {
                    suffix: b"ana".to_vec(),
                    weight: 20
                },
            ]
        );

        // The exact hit consumes one of the `n` slots, exactly as Java's
        // `if (--num == 0) return results;` does.
        assert_eq!(
            top_n_completions(&fst, b"band", 1, true).unwrap(),
            vec![Completion {
                suffix: Vec::new(),
                weight: 20
            }]
        );

        // A prefix that is not itself a term behaves identically either way.
        assert_eq!(
            top_n_completions(&fst, b"app", 2, true).unwrap(),
            top_n_completions(&fst, b"app", 2, false).unwrap()
        );

        // An exact hit with no continuations still comes back on its own.
        assert_eq!(
            top_n_completions(&fst, b"banana", 5, true).unwrap(),
            vec![Completion {
                suffix: Vec::new(),
                weight: 100
            }]
        );

        // No match at all, with `exact_first` on, is still empty.
        assert_eq!(
            top_n_completions(&fst, b"zzz", 5, true).unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn build_suggester_fst_empty_input() {
        let fst = build_suggester_fst(&[]).unwrap();
        assert_eq!(top_n_completions(&fst, b"", 5, false).unwrap(), Vec::new());
    }
}
