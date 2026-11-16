#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)] // Test arithmetic is not read off disk.
use super::*;

/// `(arrangement, startPosition(), endPosition())` -- one emitted match,
/// which is the whole observable of either walk.
type Emitted = (Vec<(i32, i32)>, i32, i32);

fn unordered(clause_spans: &[&[(i32, i32)]], slop: i64) -> Vec<Emitted> {
    let mut out = Vec::new();
    for_each_unordered_match(clause_spans, slop, |spans, start, end| {
        out.push((spans.to_vec(), start, end));
    });
    out
}

fn ordered(clause_spans: &[&[(i32, i32)]], slop: i64) -> Vec<Emitted> {
    let mut out = Vec::new();
    for_each_ordered_match(clause_spans, slop, |spans, start, end| {
        out.push((spans.to_vec(), start, end));
    });
    out
}

fn unordered_extents(clause_spans: &[&[(i32, i32)]], slop: i64) -> Vec<(i32, i32)> {
    unordered(clause_spans, slop)
        .into_iter()
        .map(|(_, s, e)| (s, e))
        .collect()
}

fn ordered_extents(clause_spans: &[&[(i32, i32)]], slop: i64) -> Vec<(i32, i32)> {
    ordered(clause_spans, slop)
        .into_iter()
        .map(|(_, s, e)| (s, e))
        .collect()
}

#[test]
fn a_transposed_adjacent_pair_has_width_zero() {
    // "beta alpha" over alpha@0 beta@1: clause 0 (beta) at 1, clause 1
    // (alpha) at 0. `max(2) - min(0) - 2 == 0`, so it matches at slop 0 --
    // where `SloppyPhraseMatcher` needs slop 2.
    assert_eq!(
        unordered_extents(&[&[(1, 2)][..], &[(0, 1)][..]], 0),
        vec![(0, 2)]
    );
    // In order, clause 1 would have to start at or after clause 0's end,
    // and it cannot: `NearSpansOrdered` never matches a transposition.
    assert!(ordered_extents(&[&[(1, 2)][..], &[(0, 1)][..]], 9).is_empty());
}

#[test]
fn overlapping_spans_give_a_negative_width_and_still_match() {
    // Two clauses on one occurrence: `1 - 0 - 2 == -1`. No non-overlap
    // rule exists in `NearSpansUnordered`.
    assert_eq!(
        unordered_extents(&[&[(0, 1)][..], &[(0, 1)][..]], 0),
        vec![(0, 1)]
    );
    // `NearSpansOrdered.stretchToOrder` requires it, so the same input
    // matches nothing there at any slop.
    assert!(ordered_extents(&[&[(0, 1)][..], &[(0, 1)][..]], 9).is_empty());
}

#[test]
fn a_gap_costs_exactly_the_intervening_positions() {
    // alpha@0, beta@3: `4 - 0 - 2 == 2`.
    assert!(unordered_extents(&[&[(0, 1)][..], &[(3, 4)][..]], 1).is_empty());
    assert_eq!(
        unordered_extents(&[&[(0, 1)][..], &[(3, 4)][..]], 2),
        vec![(0, 4)]
    );
    // In order the same gap costs `3 - 1 == 2`, the identical number: for a
    // non-overlapping arrangement `sum(next.start - prev.end)` telescopes
    // to `maxEnd - minStart - sum(lengths)`.
    assert!(ordered_extents(&[&[(0, 1)][..], &[(3, 4)][..]], 1).is_empty());
    assert_eq!(
        ordered_extents(&[&[(0, 1)][..], &[(3, 4)][..]], 2),
        vec![(0, 4)]
    );
}

#[test]
fn wider_sub_spans_pay_only_for_the_gaps_between_them() {
    // A three-position span and a one-position span, touching: width 0.
    assert_eq!(
        unordered_extents(&[&[(0, 3)][..], &[(3, 4)][..]], 0),
        vec![(0, 4)]
    );
    // One position apart: width 1, so slop 0 rejects and slop 1 accepts.
    assert!(unordered_extents(&[&[(0, 3)][..], &[(4, 5)][..]], 0).is_empty());
    assert_eq!(
        unordered_extents(&[&[(0, 3)][..], &[(4, 5)][..]], 1),
        vec![(0, 5)]
    );
}

#[test]
fn the_walk_visits_every_configuration_the_min_advance_reaches() {
    // Slot A at 0 and 10, slot B at 5. Java advances the least span each
    // step, so both configurations are visited.
    let found = unordered(&[&[(0, 1), (10, 11)][..], &[(5, 6)][..]], 4);
    assert_eq!(
        found,
        vec![
            (vec![(0, 1), (5, 6)], 0, 6),
            // `maxEndPosition` is the running maximum, so the second match
            // reports end 11 even though its own spans end at 11 -- here
            // they agree; `max_end_position_never_falls_back_once_raised`
            // is the case where they do not.
            (vec![(10, 11), (5, 6)], 5, 11),
        ]
    );
    // At slop 3 neither fits (`6-0-2 == 4`, `11-5-2 == 4`).
    assert!(unordered(&[&[(0, 1), (10, 11)][..], &[(5, 6)][..]], 3).is_empty());
}

#[test]
fn an_empty_clause_or_no_clauses_yields_nothing() {
    assert!(unordered(&[], 5).is_empty());
    assert!(unordered(&[&[(0, 1)][..], &[][..]], 5).is_empty());
    assert!(ordered(&[], 5).is_empty());
    assert!(ordered(&[&[(0, 1)][..], &[][..]], 5).is_empty());
}

#[test]
fn max_end_position_never_falls_back_once_raised() {
    // Java only ever raises `maxEndPosition`. Slot A: [0,4), [5,6).
    // Slot B: [9,10). After A moves [0,4) -> [5,6), `maxEndPosition` is
    // still 10 (B's) -- but so is the arrangement's own max end, so the
    // asymmetry needs A's *first* span to be the widest.
    let found = unordered(&[&[(0, 8), (9, 10)][..], &[(1, 2)][..]], 8);
    assert_eq!(
        found,
        vec![
            (vec![(0, 8), (1, 2)], 0, 8),
            // A has moved to [9,10) and B still holds [1,2): the
            // arrangement's own maximum end is 10, and `maxEndPosition` is
            // also 10. Advance once more and the running max stays 10.
            (vec![(9, 10), (1, 2)], 1, 10),
        ]
    );
    // The running-maximum case proper: the widest span is passed *before*
    // a later match, so the reported end exceeds every held span's end.
    // Slot A: [0,20), [21,22). Slot B: [21,22).
    let found = unordered(&[&[(0, 20), (21, 22)][..], &[(21, 22)][..]], 30);
    assert_eq!(
        found,
        vec![
            (vec![(0, 20), (21, 22)], 0, 22),
            // Both clauses now hold [21,22) -- maximum end 22 -- yet Java
            // reports `maxEndPosition == 22` here too. Raise A's first span
            // past B's to separate them.
            (vec![(21, 22), (21, 22)], 21, 22),
        ]
    );
}

#[test]
fn the_reported_end_is_the_running_maximum_not_the_arrangements_own() {
    // Clause 0: [0,50) then [60,61). Clause 1: [60,61).
    // Step 1: top is clause 0 at 0; maxEnd = 61; width 61-0-51 = 10.
    // Step 2: clause 0 advances to [60,61); both clauses hold [60,61);
    //         maxEnd stays 61 and the arrangement's own max end is 61.
    // To make them differ the *advancing* span must be the one that had
    // set the maximum, and the new arrangement must end earlier -- which
    // is exactly clause 1 sitting below clause 0's old end.
    let found = unordered(&[&[(0, 50), (55, 56)][..], &[(51, 52)][..]], 60);
    assert_eq!(
        found,
        vec![
            (vec![(0, 50), (51, 52)], 0, 52),
            // Clause 0 moved off [0,50) to [55,56). The arrangement now
            // spans [51,56), whose own maximum end is 56 -- but Java's
            // `endPosition()` is the running `maxEndPosition`, still 56.
            (vec![(55, 56), (51, 52)], 51, 56),
        ]
    );
}

#[test]
fn three_clauses_use_the_widest_window_not_the_summed_gaps() {
    // a@0, b@5, c@1: `6 - 0 - 3 == 3`.
    assert!(unordered_extents(&[&[(0, 1)][..], &[(5, 6)][..], &[(1, 2)][..]], 2).is_empty());
    assert_eq!(
        unordered_extents(&[&[(0, 1)][..], &[(5, 6)][..], &[(1, 2)][..]], 3),
        vec![(0, 6)]
    );
}

#[test]
fn the_ordered_walk_emits_only_minimum_slop_matches() {
    // `alpha@0`, `beta@1`, `beta@5` with a budget of 5. The greedy
    // `stretchToOrder` advance settles beta on 1 and never revisits it, so
    // `[0, 6)` -- which a cartesian product accepts, width 4 -- is never
    // emitted. The class doc says exactly this: "the formed spans only
    // contains minimum slop matches".
    assert_eq!(
        ordered_extents(&[&[(0, 1)][..], &[(1, 2), (5, 6)][..]], 5),
        vec![(0, 2)]
    );
}

#[test]
fn the_ordered_walk_reaches_the_second_beta_once_alpha_moves_past_the_first() {
    // The javadoc's own example, one clause pair at a time: `t1 t2 t1 t3`.
    // alpha at 0 and 2, beta at 1 and 3.
    assert_eq!(
        ordered_extents(&[&[(0, 1), (2, 3)][..], &[(1, 2), (3, 4)][..]], 0),
        vec![(0, 2), (2, 4)]
    );
}

#[test]
fn an_exhausted_later_clause_ends_the_ordered_walk() {
    // Clause 1 has one span, at 1. Once clause 0 passes it nothing can
    // follow, and Java stops the document rather than trying clause 0's
    // remaining positions.
    assert_eq!(
        ordered_extents(&[&[(0, 1), (5, 6), (7, 8)][..], &[(1, 2)][..]], 9),
        vec![(0, 2)]
    );
}

#[test]
fn the_ordered_walk_rejects_an_over_budget_arrangement_and_keeps_going() {
    // alpha at 0 and 4; beta at 3 only. From alpha@0 the width is
    // `3 - 1 == 2`, over a slop of 1; from alpha@4 beta is exhausted. So
    // nothing is emitted, and the walk terminates.
    assert!(ordered_extents(&[&[(0, 1), (4, 5)][..], &[(3, 4)][..]], 1).is_empty());
    // With the budget raised the same first arrangement matches.
    assert_eq!(
        ordered_extents(&[&[(0, 1), (4, 5)][..], &[(3, 4)][..]], 2),
        vec![(0, 4)]
    );
}

#[test]
fn a_single_clause_emits_every_span_on_both_arms() {
    // `stretchToOrder`'s loop body never runs, so `matchWidth` is 0 and
    // every span of the lone clause is a match; the unordered arm's window
    // is `end - start - (end - start) == 0` likewise.
    let spans: &[(i32, i32)] = &[(0, 1), (4, 5), (9, 10)];
    assert_eq!(ordered_extents(&[spans], 0), vec![(0, 1), (4, 5), (9, 10)]);
    assert_eq!(
        unordered_extents(&[spans], 0),
        vec![(0, 1), (4, 5), (9, 10)]
    );
}

#[test]
fn three_ordered_clauses_advance_independently() {
    // a@{0,6}, b@{2,7}, c@{4,9}: from a@0 the greedy walk takes b@2, c@4
    // (width (2-1) + (4-3) = 2); from a@6 it takes b@7, c@9 (width 1 + 1).
    assert_eq!(
        ordered_extents(
            &[
                &[(0, 1), (6, 7)][..],
                &[(2, 3), (7, 8)][..],
                &[(4, 5), (9, 10)][..],
            ],
            2
        ),
        vec![(0, 5), (6, 10)]
    );
}
