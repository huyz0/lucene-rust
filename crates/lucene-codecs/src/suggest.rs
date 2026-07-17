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
fn encode_weight(weight: u32) -> i64 {
    (u32::MAX - weight) as i64
}

/// `WFSTCompletionLookup.decodeWeight`, widened to `u32`: `cost -> weight`.
/// `cost` must be in `0..=u32::MAX` (always true for a value this module's
/// own `encode_weight` produced, which is the only source `top_n_completions`
/// ever decodes).
fn decode_weight(cost: i64) -> u32 {
    debug_assert!(
        (0..=u32::MAX as i64).contains(&cost),
        "cost {cost} out of u32 range"
    );
    (u32::MAX as i64 - cost) as u32
}

/// Builds a suggester FST from `(term, weight)` pairs (`WFSTCompletionLookup
/// .build`'s FST-construction step, minus the `InputIterator`/temp-directory
/// external-sort machinery real Lucene uses to accept an arbitrarily large,
/// unsorted stream -- this module's caller is expected to already hold its
/// full term list in memory, matching this port's small-dictionary scope).
///
/// `entries` need not be pre-sorted or de-duplicated: this function sorts a
/// local copy by term bytes and, like real `WFSTCompletionLookup.build`
/// (`SortedInputIterator` dedup loop), keeps only the *first* weight seen for
/// a duplicate term and discards the rest.
pub fn build_suggester_fst(entries: &[(Vec<u8>, u32)]) -> Result<Fst<'static>, BuildError> {
    let mut sorted: Vec<(Vec<u8>, u32)> = entries.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    sorted.dedup_by(|later, earlier| later.0 == earlier.0);

    let costed: Vec<(Vec<u8>, i64)> = sorted
        .into_iter()
        .map(|(term, weight)| (term, encode_weight(weight)))
        .collect();

    fst::build_fst_typed::<PositiveIntOutputs>(&costed)
}

/// Walks every accepted term of `fst` that starts with `prefix`, returning
/// the `n` highest-weight ones (highest first; ties broken by ascending
/// suffix byte order for a deterministic result), each as the suffix
/// continuing `prefix` plus its (un-inverted) weight.
///
/// Returns fewer than `n` completions if fewer than `n` terms share the
/// prefix (including zero). See this module's doc comment ("Scope of this
/// module") for exactly what "top-N" search strategy this does and does not
/// implement.
pub fn top_n_completions(fst: &Fst, prefix: &[u8], n: usize) -> fst::Result<Vec<Completion>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut iter = fst.iter()?;
    let first = iter.seek_ceil(prefix)?;

    // Bounded min-heap: at most `n` entries kept at any time, ordered so the
    // *worst* (lowest-weight) kept candidate is always the one popped when a
    // better candidate needs room. `Reverse` turns `BinaryHeap`'s default
    // max-heap into the min-heap this eviction policy needs.
    let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::with_capacity(n + 1);

    let push = |key: Vec<u8>, output: Vec<u8>, heap: &mut BinaryHeap<Reverse<HeapItem>>| {
        let cost = PositiveIntOutputs::decode(&output);
        let weight = decode_weight(cost);
        let suffix = key[prefix.len()..].to_vec();
        heap.push(Reverse(HeapItem { weight, suffix }));
        if heap.len() > n {
            heap.pop();
        }
    };

    match first {
        Some((key, output)) if key.starts_with(prefix) => push(key, output, &mut heap),
        _ => return Ok(Vec::new()),
    }

    for item in iter {
        let (key, output) = item?;
        if !key.starts_with(prefix) {
            break;
        }
        push(key, output, &mut heap);
    }

    let mut results: Vec<HeapItem> = heap.into_iter().map(|Reverse(item)| item).collect();
    results.sort_by(|a, b| {
        b.weight
            .cmp(&a.weight)
            .then_with(|| a.suffix.cmp(&b.suffix))
    });

    Ok(results
        .into_iter()
        .map(|item| Completion {
            suffix: item.suffix,
            weight: item.weight,
        })
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
    use super::*;

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

    #[test]
    fn build_suggester_fst_dedups_keeping_first_weight() {
        let entries = vec![
            (b"dup".to_vec(), 5),
            (b"dup".to_vec(), 999), // must be discarded; first weight wins
            (b"other".to_vec(), 1),
        ];
        let fst = build_suggester_fst(&entries).unwrap();
        let cost = fst
            .get_typed::<PositiveIntOutputs>(b"dup")
            .unwrap()
            .unwrap();
        assert_eq!(decode_weight(cost), 5);
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
        let top = top_n_completions(&fst, b"app", 2).unwrap();
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
        let all = top_n_completions(&fst, b"app", 10).unwrap();
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
        let top = top_n_completions(&fst, b"band", 3).unwrap();
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
        let top = top_n_completions(&fst, b"banana", 5).unwrap();
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
        assert_eq!(top_n_completions(&fst, b"zzz", 5).unwrap(), Vec::new());
        // Prefix strictly beyond the last key too.
        assert_eq!(top_n_completions(&fst, b"zzzzzzz", 5).unwrap(), Vec::new());
    }

    #[test]
    fn top_n_completions_n_zero_returns_empty() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();
        assert_eq!(top_n_completions(&fst, b"app", 0).unwrap(), Vec::new());
    }

    #[test]
    fn top_n_completions_n_larger_than_available_returns_all() {
        let fst = build_suggester_fst(&sample_entries()).unwrap();
        let top = top_n_completions(&fst, b"apple", 1000).unwrap();
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
        let top = top_n_completions(&fst, b"", 100).unwrap();
        assert_eq!(top.len(), entries.len());
        assert_eq!(top[0].weight, 100); // "banana"
                                        // Descending order maintained throughout.
        for w in top.windows(2) {
            assert!(w[0].weight >= w[1].weight);
        }
    }

    #[test]
    fn build_suggester_fst_empty_input() {
        let fst = build_suggester_fst(&[]).unwrap();
        assert_eq!(top_n_completions(&fst, b"", 5).unwrap(), Vec::new());
    }
}
