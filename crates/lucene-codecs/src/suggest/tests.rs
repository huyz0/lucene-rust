#![cfg(test)]
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
    let fst =
        crate::fst::build_fst_typed::<PositiveIntOutputs>(&[(b"ap".to_vec(), i64::MAX)]).unwrap();
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
fn top_n_completions_keeps_the_smaller_suffix_on_a_tie_at_the_cut_off() {
    let fst = build_suggester_fst(&sample_entries()).unwrap();
    // ""(20) and "ana"(20) tie for the second place: Java keeps "".
    let top = top_n_completions(&fst, b"band", 2, false).unwrap();
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
