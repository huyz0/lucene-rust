//! `NearSpansOrdered` and `NearSpansUnordered`
//! (`/home/tuong/work/lucene-10.5.0/lucene/queries/src/java/org/apache/lucene/queries/spans/NearSpans{Ordered,Unordered}.java`),
//! over already-decoded spans.
//!
//! ## Why this is not `sloppy_phrase`
//!
//! Two different "unordered phrase within a slop budget" machines live in
//! Lucene, they disagree, and which one applies depends on *who is asking*:
//!
//! - **Scoring** a sloppy `PhraseQuery` uses `SloppyPhraseMatcher`
//!   ([`crate::sloppy_phrase`]), whose `matchLength` is a window width over
//!   *slot-shifted* positions: `max(p_i - i) - min(p_i - i)`.
//! - **Highlighting** the same query uses `SpanNearQuery`.
//!   `WeightedSpanTermExtractor.extract` rewrites a `PhraseQuery` as
//!   ```java
//!   boolean inorder = (phraseQuery.getSlop() == 0);
//!   new SpanNearQuery(clauses, phraseQuery.getSlop() + positionGaps, inorder)
//!   ```
//!   and `NearSpansUnordered.atMatch()` is
//!   `maxEndPosition - top().startPosition() - totalSpanLength <= slop`,
//!   i.e. `max(p) - min(p) + 1 - n` for `n` one-position term spans.
//!
//! For the reordered pair `alpha@0 beta@1` queried as `"beta alpha"` those
//! two quantities are **0** and **2**: real Lucene highlights the span at slop
//! 1 while the scorer does not match the document until slop 2, and at slop 0
//! the highlighter does not highlight at all (because `inorder` is forced
//! there). All three are recorded, from real Lucene, as
//! `highlight.reordered_slop{0,1,2}` in
//! `fixtures/data/blocktree_index/manifest.properties`.
//!
//! That is why "make the highlighter enumerate through `SloppyPhraseMatcher`"
//! -- the obvious reading, and the one this port's own ledger recorded --
//! would have been a *different* wrong answer rather than a fix.
//!
//! ## What `atMatch` does and does not require
//!
//! It is a pure width test. `NearSpansUnordered` has **no** non-overlap
//! requirement and **no** repeated-term handling: two clauses holding the same
//! term may settle on the same position, giving a negative width, which
//! matches. Real Lucene therefore highlights `"alpha alpha"~2` in a document
//! containing a single `alpha` (`highlight.repeat_single_occurrence`), a
//! document the *query* does not match. `NearSpansOrdered` is the opposite: it
//! advances each sub-span to `>= prevSpans.endPosition()`, so non-overlap is
//! required there.
//!
//! ## Why both arms are walks and not a cartesian product
//!
//! Both Java classes are `Spans` iterators, and the *sequence* they emit is
//! narrower than "every arrangement satisfying the budget" -- which is what
//! this port enumerated until `c43-final-cleanup`, and which reported extents
//! Lucene never produces:
//!
//! - `NearSpansOrdered`'s sub-span cursors are **monotonic**. `stretchToOrder`
//!   advances each later clause with `while (spans.startPosition() < position)
//!   spans.nextStartPosition()` and never rewinds, so for each position of
//!   clause 0 there is exactly *one* arrangement -- the **minimum-slop** one.
//!   The class doc says so: "the formed spans only contains minimum slop
//!   matches". A document with `alpha@0`, `beta@1`, `beta@5` yields
//!   `[0, 2)` and nothing else for `SpanNear([alpha, beta], 5, true)`; the
//!   cartesian product also yields `[0, 6)`.
//! - `NearSpansUnordered` advances only the queue's least element, so it visits
//!   a single path through the arrangement lattice rather than all of it. And
//!   its `endPosition()` is `spanWindow.maxEndPosition`, the **running** max,
//!   which can exceed the current arrangement's own maximum end.
//!
//! Neither difference changes a *hit set* -- [`crate::span_doc_ids`] only asks
//! whether the span list is non-empty, and a walk that emits nothing is a walk
//! over an arrangement lattice that contains no match. It changes the extents,
//! which a nested `SpanNear`-of-`SpanNear` consumes.

/// One arrangement the walk visits: the span each clause is currently
/// positioned on, in clause order.
///
/// This is what `NearSpans*.collect(SpanCollector)` hands the collector
/// (`for (Spans spans : subSpans) spans.collect(collector)`), and it is what
/// the highlighter needs -- the *leaf* occurrences taking part in the match,
/// not just the overall extent.
pub type Arrangement<'a> = &'a [(i32, i32)];

/// `NearSpansOrdered`'s enumeration: `twoPhaseCurrentDocMatches()` followed by
/// `nextStartPosition()` until `NO_MORE_POSITIONS`.
///
/// ```java
/// while (subSpans[0].nextStartPosition() != NO_MORE_POSITIONS && !oneExhaustedInCurrentDoc) {
///   if (stretchToOrder() && matchWidth <= allowedSlop) {
///     return matchStart;
///   }
/// }
/// ```
///
/// `clause_spans[i]` is clause `i`'s spans in this one document, ascending by
/// `(start, end)` -- the order `Spans.nextStartPosition()` yields them in.
/// `on_match` is called with `(arrangement, matchStart, matchEnd)`, where
/// `matchStart` is `subSpans[0].startPosition()` and `matchEnd` is the *last*
/// clause's `endPosition()`, exactly as `stretchToOrder` sets them.
///
/// **The cursors never rewind**, which is the whole difference from a
/// cartesian product: `advancePosition` only ever calls `nextStartPosition()`,
/// so once clause `i` has passed a span it is gone for every later position of
/// clause 0 as well. That is what makes each emitted arrangement the
/// minimum-slop one for its `matchStart`, and it is why the walk is linear in
/// the total number of spans rather than exponential in the clause count.
///
/// **One exhausted sub-span ends the document**, not just the current
/// candidate: `stretchToOrder` sets `oneExhaustedInCurrentDoc` and the `while`
/// condition then stops the loop. Since every clause's cursor is monotonic and
/// `prevSpans.endPosition()` only rises with clause 0's position, a clause that
/// cannot be advanced far enough now can never be advanced far enough later, so
/// this is an early exit rather than a lost match.
pub fn for_each_ordered_match(
    clause_spans: &[&[(i32, i32)]],
    slop: i64,
    mut on_match: impl FnMut(Arrangement<'_>, i32, i32),
) {
    let n = clause_spans.len();
    if n == 0 || clause_spans.iter().any(|s| s.is_empty()) {
        return;
    }
    // `unpositioned()`: every sub-span at `startPosition() == -1`, which is
    // below every real position, so the first `advancePosition` moves it onto
    // its first span.
    let mut cursor: Vec<Option<usize>> = vec![None; n];
    let mut current: Vec<(i32, i32)> = vec![(0, 0); n];
    loop {
        // `subSpans[0].nextStartPosition() != NO_MORE_POSITIONS`.
        let first = match cursor[0] {
            None => 0,
            Some(at) => at.saturating_add(1),
        };
        let Some(&(match_start, first_end)) = clause_spans[0].get(first) else {
            return;
        };
        cursor[0] = Some(first);
        current[0] = (match_start, first_end);

        // `stretchToOrder()`.
        let mut prev_end = first_end;
        let mut width: i64 = 0;
        let mut exhausted = false;
        for i in 1..n {
            // `advancePosition(spans, prevSpans.endPosition())`:
            // `while (spans.startPosition() < position) spans.nextStartPosition();`
            let mut at = cursor[i];
            while at.map_or(-1, |c| clause_spans[i][c].0) < prev_end {
                let next = at.map_or(0, |c| c.saturating_add(1));
                if next >= clause_spans[i].len() {
                    // `NO_MORE_POSITIONS`: `oneExhaustedInCurrentDoc = true`.
                    exhausted = true;
                    break;
                }
                at = Some(next);
            }
            if exhausted {
                break;
            }
            let span = clause_spans[i][at.expect("advanced onto a span")];
            cursor[i] = at;
            current[i] = span;
            // `matchWidth += (spans.startPosition() - prevSpans.endPosition())`.
            width = width.saturating_add(i64::from(span.0) - i64::from(prev_end));
            prev_end = span.1;
        }
        if exhausted {
            return;
        }
        // `matchEnd = subSpans[subSpans.length - 1].endPosition()`.
        if width <= slop {
            on_match(&current, match_start, prev_end);
        }
    }
}

/// `NearSpansUnordered`'s own enumeration: `twoPhaseCurrentDocMatches()`
/// followed by `nextStartPosition()` until `NO_MORE_POSITIONS`, calling
/// `on_match` with the span each clause is currently positioned on -- which is
/// what `NearSpansUnordered.collect(SpanCollector)` hands to the collector
/// (`for (Spans spans : subSpans) spans.collect(collector)`).
///
/// `clause_spans[i]` is clause `i`'s spans in this one document, ascending by
/// `(start, end)` -- the order `Spans.nextStartPosition()` yields them in.
///
/// The walk is Lucene's: a priority queue ordered by
/// `positionsOrdered` (start, then end), advance the least element, recheck.
///
/// `on_match` receives `(arrangement, startPosition(), endPosition())`, and
/// those last two are **not** the arrangement's own minimum start and maximum
/// end. Java's are
///
/// ```java
/// public int startPosition() { return spanWindow.top().startPosition(); }
/// public int endPosition()   { return spanWindow.maxEndPosition; }
/// ```
///
/// -- and `maxEndPosition` is a *running* maximum that only ever rises, never
/// recomputed when the span that set it moves on. So a match reached after a
/// long sub-span has been passed reports an end position beyond every span it
/// actually holds. That is Lucene's extent, and reproducing it is the point:
/// an outer `SpanNear` clause consumes these extents.
///
/// `atMatch()` is `maxEndPosition - top().startPosition() - totalSpanLength <=
/// allowedSlop`, where `totalSpanLength` is likewise maintained incrementally
/// (`nextPosition()` subtracts the advancing span's old width and adds its
/// new one), so it always describes the current arrangement even though
/// `maxEndPosition` does not. Both are widened to `i64` here because every
/// input is a position read off a `.pos` file and the three-term subtraction is
/// unbounded below.
///
/// **One deliberate difference**: `positionsOrdered` is not a total order (two
/// spans with the same start *and* end compare "not less" both ways), so
/// **Java's answer for a tie is unspecified** -- which of them a binary heap
/// makes `top()` depends on its internal array order. The clause index breaks
/// the tie here, which makes the walk deterministic.
///
/// It is not a difference without consequence, and the honest statement is
/// that no port can match an unspecified order rather than that the choice
/// cannot matter. `top()` is also the span that gets *advanced*, so the tie
/// selects a different continuation: clause 0 at `[(0,1), (2,3)]` and clause 1
/// at `[(0,1), (9,10)]` with slop 1 reaches the matching `{(2,3), (0,1)}` if
/// clause 0 moves first and never reaches it if clause 1 does. Two clauses can
/// only tie on `(start, end)` when two *different* terms share a position (a
/// synonym injected at `position_increment == 0`); two slots holding the *same*
/// term share one span list, so their continuations are symmetric and the tie
/// really is immaterial there.
pub fn for_each_unordered_match(
    clause_spans: &[&[(i32, i32)]],
    slop: i64,
    mut on_match: impl FnMut(Arrangement<'_>, i32, i32),
) {
    let n = clause_spans.len();
    if n == 0 || clause_spans.iter().any(|s| s.is_empty()) {
        return;
    }
    // `startDocument()`: every sub-span on its first position.
    let mut cursor = vec![0usize; n];
    let mut current: Vec<(i32, i32)> = clause_spans.iter().map(|s| s[0]).collect();
    let mut max_end: i32 = current
        .iter()
        .map(|&(_, end)| end)
        .max()
        .expect("non-empty: n > 0");
    // `total_length` is fixed only while every clause's spans are the same
    // width; Java recomputes it per advance, so do the same.
    let mut total_length: i64 = current
        .iter()
        .map(|&(start, end)| i64::from(end) - i64::from(start))
        .sum();
    // The queue, as clause indices sorted by `(start, end, clause)`.
    let mut queue: Vec<usize> = (0..n).collect();
    queue.sort_by_key(|&i| (current[i].0, current[i].1, i));

    loop {
        let top = queue[0];
        // `atMatch()`.
        if i64::from(max_end)
            .saturating_sub(i64::from(current[top].0))
            .saturating_sub(total_length)
            <= slop
        {
            on_match(&current, current[top].0, max_end);
        }
        // `nextPosition()`: advance the least span; the walk ends when it is
        // exhausted, exactly as Java's does (one exhausted sub-span ends the
        // document, however many positions the others have left).
        let old = current[top];
        cursor[top] = cursor[top].saturating_add(1);
        let Some(&next) = clause_spans[top].get(cursor[top]) else {
            return;
        };
        current[top] = next;
        total_length = total_length
            .saturating_sub(i64::from(old.1) - i64::from(old.0))
            .saturating_add(i64::from(next.1) - i64::from(next.0));
        max_end = max_end.max(next.1);
        // `updateTop()`.
        queue.remove(0);
        let key = (next.0, next.1, top);
        let at = queue.partition_point(|&j| (current[j].0, current[j].1, j) < key);
        queue.insert(at, top);
    }
}

#[cfg(test)]
mod tests {
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
}
