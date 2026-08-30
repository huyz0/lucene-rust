//! `NearSpansUnordered`
//! (`/home/tuong/work/lucene-10.5.0/lucene/queries/src/java/org/apache/lucene/queries/spans/NearSpansUnordered.java`),
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

/// `SpanTotalLengthEndPositionWindow.atMatch()`:
/// `maxEndPosition - top().startPosition() - totalSpanLength <= allowedSlop`,
/// where `top()` is the span with the smallest start.
///
/// `spans` is one chosen `[start, end)` per clause, in any order. Widened to
/// `i64` because every input is a position read off a `.pos` file and the
/// three-term subtraction is unbounded below.
pub fn unordered_width(spans: &[(i32, i32)]) -> i64 {
    let mut min_start = i64::MAX;
    let mut max_end = i64::MIN;
    let mut total_length: i64 = 0;
    for &(start, end) in spans {
        min_start = min_start.min(i64::from(start));
        max_end = max_end.max(i64::from(end));
        total_length = total_length.saturating_add(i64::from(end) - i64::from(start));
    }
    max_end
        .saturating_sub(min_start)
        .saturating_sub(total_length)
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
/// `maxEndPosition` only ever rises, exactly as Java's does -- it is not
/// recomputed when the span that set it moves on.
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
    mut on_match: impl FnMut(&[(i32, i32)]),
) {
    let n = clause_spans.len();
    if n == 0 || clause_spans.iter().any(|s| s.is_empty()) {
        return;
    }
    // `startDocument()`: every sub-span on its first position.
    let mut cursor = vec![0usize; n];
    let mut current: Vec<(i32, i32)> = clause_spans.iter().map(|s| s[0]).collect();
    let mut max_end: i64 = current
        .iter()
        .map(|&(_, end)| i64::from(end))
        .max()
        .unwrap_or(i64::MIN);
    let total_length: i64 = current
        .iter()
        .map(|&(start, end)| i64::from(end) - i64::from(start))
        .sum();
    // The queue, as clause indices sorted by `(start, end, clause)`.
    let mut queue: Vec<usize> = (0..n).collect();
    queue.sort_by_key(|&i| (current[i].0, current[i].1, i));

    // `total_length` is fixed only while every clause's spans are the same
    // width; Java recomputes it per advance, so do the same.
    let mut total_length = total_length;
    loop {
        let top = queue[0];
        if max_end
            .saturating_sub(i64::from(current[top].0))
            .saturating_sub(total_length)
            <= slop
        {
            on_match(&current);
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
        max_end = max_end.max(i64::from(next.1));
        // `updateTop()`.
        queue.remove(0);
        let key = (next.0, next.1, top);
        let at = queue.partition_point(|&j| (current[j].0, current[j].1, j) < key);
        queue.insert(at, top);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(clause_spans: &[&[(i32, i32)]], slop: i64) -> Vec<Vec<(i32, i32)>> {
        let mut out = Vec::new();
        for_each_unordered_match(clause_spans, slop, |spans| out.push(spans.to_vec()));
        out
    }

    #[test]
    fn a_transposed_adjacent_pair_has_width_zero() {
        // "beta alpha" over alpha@0 beta@1: clause 0 (beta) at 1, clause 1
        // (alpha) at 0. `max(2) - min(0) - 2 == 0`, so it matches at slop 0 --
        // where `SloppyPhraseMatcher` needs slop 2.
        assert_eq!(unordered_width(&[(1, 2), (0, 1)]), 0);
        assert_eq!(matches(&[&[(1, 2)][..], &[(0, 1)][..]], 0).len(), 1);
    }

    #[test]
    fn overlapping_spans_give_a_negative_width_and_still_match() {
        // Two clauses on one occurrence: `1 - 0 - 2 == -1`. No non-overlap
        // rule exists in `NearSpansUnordered`.
        assert_eq!(unordered_width(&[(0, 1), (0, 1)]), -1);
        assert_eq!(matches(&[&[(0, 1)][..], &[(0, 1)][..]], 0).len(), 1);
    }

    #[test]
    fn a_gap_costs_exactly_the_intervening_positions() {
        // alpha@0, beta@3: `4 - 0 - 2 == 2`.
        assert_eq!(unordered_width(&[(0, 1), (3, 4)]), 2);
        assert!(matches(&[&[(0, 1)][..], &[(3, 4)][..]], 1).is_empty());
        assert_eq!(matches(&[&[(0, 1)][..], &[(3, 4)][..]], 2).len(), 1);
    }

    #[test]
    fn wider_sub_spans_pay_only_for_the_gaps_between_them() {
        // A three-position span and a one-position span, touching: width 0.
        assert_eq!(unordered_width(&[(0, 3), (3, 4)]), 0);
        // One position apart: width 1.
        assert_eq!(unordered_width(&[(0, 3), (4, 5)]), 1);
    }

    #[test]
    fn the_walk_visits_every_configuration_the_min_advance_reaches() {
        // Slot A at 0 and 10, slot B at 5. Java advances the least span each
        // step, so both configurations are visited.
        let found = matches(&[&[(0, 1), (10, 11)][..], &[(5, 6)][..]], 4);
        assert_eq!(found, vec![vec![(0, 1), (5, 6)], vec![(10, 11), (5, 6)]]);
        // At slop 3 neither fits (`6-0-2 == 4`, `11-5-2 == 4`).
        assert!(matches(&[&[(0, 1), (10, 11)][..], &[(5, 6)][..]], 3).is_empty());
    }

    #[test]
    fn an_empty_clause_or_no_clauses_yields_nothing() {
        assert!(matches(&[], 5).is_empty());
        assert!(matches(&[&[(0, 1)][..], &[][..]], 5).is_empty());
    }

    #[test]
    fn max_end_position_never_falls_back_once_raised() {
        // Java only ever raises `maxEndPosition`. Slot A: 0, 1. Slot B: 9.
        // After A moves 0 -> 1, `maxEndPosition` is still 10 (B's), and the
        // window is 10 - 1 - 2 = 7.
        let found = matches(&[&[(0, 1), (1, 2)][..], &[(9, 10)][..]], 7);
        assert_eq!(found, vec![vec![(1, 2), (9, 10)]]);
    }

    #[test]
    fn three_clauses_use_the_widest_window_not_the_summed_gaps() {
        // a@0, b@5, c@1: `6 - 0 - 3 == 3`.
        assert_eq!(unordered_width(&[(0, 1), (5, 6), (1, 2)]), 3);
        assert!(matches(&[&[(0, 1)][..], &[(5, 6)][..], &[(1, 2)][..]], 2).is_empty());
        assert_eq!(
            matches(&[&[(0, 1)][..], &[(5, 6)][..], &[(1, 2)][..]], 3).len(),
            1
        );
    }
}
