#![cfg(test)]
use super::*;

fn terms(slots: &[&str]) -> Vec<Vec<Vec<u8>>> {
    slots
        .iter()
        .map(|s| vec![s.as_bytes().to_vec()])
        .collect::<Vec<_>>()
}

fn repeats_for(slots: &[&str]) -> PhraseRepeats {
    let owned = terms(slots);
    let refs: Vec<&[Vec<u8>]> = owned.iter().map(|s| s.as_slice()).collect();
    PhraseRepeats::detect(&refs)
}

fn matches(positions: &[&[i32]], slop: u32) -> bool {
    sloppy_phrase_matches(positions, &PhraseRepeats::none(positions.len()), slop)
}

fn freq(positions: &[&[i32]], slop: u32) -> f32 {
    sloppy_phrase_freq(positions, &PhraseRepeats::none(positions.len()), slop)
}

#[test]
fn a_transposition_matches_at_slop_two_and_not_at_slop_one() {
    // The whole point of this module. Query "beta alpha" against a document
    // reading "alpha beta": beta is at 1, alpha at 0.
    assert!(!matches(&[&[1][..], &[0][..]], 1));
    assert!(matches(&[&[1][..], &[0][..]], 2));
    // Its match length is 2, so it weighs 1/3.
    assert_eq!(freq(&[&[1][..], &[0][..]], 2), 1.0 / 3.0);
}

#[test]
fn an_in_order_gap_costs_the_gap_and_a_transposition_costs_twice_the_gap_plus_two() {
    // "alpha beta" over alpha@0 beta@3: two extra words, match length 2.
    assert!(!matches(&[&[0][..], &[3][..]], 1));
    assert!(matches(&[&[0][..], &[3][..]], 2));
    // The same pair queried the other way round: beta@3 alpha@0 shifts to
    // 3 and -1, a window of 4.
    assert!(!matches(&[&[3][..], &[0][..]], 3));
    assert!(matches(&[&[3][..], &[0][..]], 4));
}

#[test]
fn adjacent_terms_match_at_slop_zero_and_weigh_one() {
    assert!(matches(&[&[0][..], &[1][..]], 0));
    assert_eq!(freq(&[&[0][..], &[1][..]], 0), 1.0);
    assert!(!matches(&[&[0][..], &[2][..]], 0));
}

#[test]
fn a_three_term_phrase_uses_the_window_width_not_the_summed_gaps() {
    // a@0 b@5 c@1: shifted to 0, 4, -1 -- window 5. The in-order reading of
    // the same document could not match at any slop, since c precedes b.
    assert!(!matches(&[&[0][..], &[5][..], &[1][..]], 4));
    assert!(matches(&[&[0][..], &[5][..], &[1][..]], 5));
}

#[test]
fn repeated_terms_at_equal_query_positions_can_share_an_occurrence() {
    // Java: SloppyPhraseMatcher.gatherRptGroups excludes equal offsets.
    // GenSimpleTextPhraseScorer also checks this through native postings.
    let terms = vec![b"a".to_vec(), b"a".to_vec()];
    let positions = [&[0][..], &[0][..]];
    let mut scratch = SloppyScratch::default();
    let shared = PhraseRepeats::for_phrase_positions(&terms, &[0, 0]);
    assert_eq!(shared.groups, [-1, -1]);
    assert_eq!(
        sloppy_phrase_freq_offsets_in(&mut scratch, &positions, &shared, 1, &[0, 0]),
        1.0
    );
    let distinct = PhraseRepeats::for_phrase_positions(&terms, &[0, 1]);
    assert_eq!(distinct.groups, [0, 0]);
    assert_eq!(
        sloppy_phrase_freq_offsets_in(&mut scratch, &positions, &distinct, 1, &[0, 1]),
        0.0
    );
    let mixed = PhraseRepeats::for_phrase_positions(
        &[b"a".to_vec(), b"a".to_vec(), b"a".to_vec()],
        &[0, 0, 1],
    );
    assert_eq!(mixed.groups, [0, -1, 0]);
    let unique = PhraseRepeats::for_phrase_positions(&[b"a".to_vec(), b"b".to_vec()], &[0, 0]);
    assert_eq!(unique, PhraseRepeats::none(2));
}

#[test]
fn repeated_terms_may_not_share_a_position() {
    let rpt = repeats_for(&["a", "a"]);
    // One occurrence of `a` cannot satisfy both slots at any slop.
    assert!(!sloppy_phrase_matches(&[&[0][..], &[0][..]], &rpt, 100));
    // Two occurrences can.
    assert!(sloppy_phrase_matches(&[&[0, 1][..], &[0, 1][..]], &rpt, 0));
    assert_eq!(
        sloppy_phrase_freq(&[&[0, 1][..], &[0, 1][..]], &rpt, 0),
        1.0
    );
    // Without the repeat information the same input matches at slop 1,
    // which is exactly the wrong answer the grouping prevents.
    assert!(matches(&[&[0][..], &[0][..]], 1));
}

#[test]
fn detect_finds_repeats_only_when_a_term_occurs_twice() {
    assert_eq!(repeats_for(&["a", "b"]), PhraseRepeats::none(2));
    let r = repeats_for(&["a", "b", "a"]);
    assert!(r.has_rpts);
    assert!(!r.has_multi_term_rpts);
    assert_eq!(r.groups, vec![0, -1, 0]);
    let r = repeats_for(&["a", "b", "a", "b"]);
    assert_eq!(r.groups, vec![0, 1, 0, 1]);
}

#[test]
fn detect_unions_multi_term_slots_that_share_a_term() {
    // MultiPhraseQuery shape: slot 0 = {a, b}, slot 1 = {b, c}, slot 2 = {d}.
    // `b` repeats, so slots 0 and 1 are one group and slot 2 is in none.
    let owned: Vec<Vec<Vec<u8>>> = vec![
        vec![b"a".to_vec(), b"b".to_vec()],
        vec![b"b".to_vec(), b"c".to_vec()],
        vec![b"d".to_vec()],
    ];
    let refs: Vec<&[Vec<u8>]> = owned.iter().map(|s| s.as_slice()).collect();
    let r = PhraseRepeats::detect(&refs);
    assert!(r.has_rpts);
    assert!(r.has_multi_term_rpts);
    assert_eq!(r.groups, vec![0, 0, -1]);
}

#[test]
fn detect_unions_transitively_across_shared_terms() {
    // {a,b} {b,c} {c,d}: a chain, so all three end up in one group even
    // though slots 0 and 2 share no term directly.
    let owned: Vec<Vec<Vec<u8>>> = vec![
        vec![b"a".to_vec(), b"b".to_vec()],
        vec![b"b".to_vec(), b"c".to_vec()],
        vec![b"c".to_vec(), b"d".to_vec()],
    ];
    let refs: Vec<&[Vec<u8>]> = owned.iter().map(|s| s.as_slice()).collect();
    let r = PhraseRepeats::detect(&refs);
    assert_eq!(r.groups, vec![0, 0, 0]);
}

#[test]
fn multi_term_repeat_groups_still_keep_slots_off_one_position() {
    let owned: Vec<Vec<Vec<u8>>> = vec![
        vec![b"a".to_vec(), b"b".to_vec()],
        vec![b"b".to_vec(), b"c".to_vec()],
    ];
    let refs: Vec<&[Vec<u8>]> = owned.iter().map(|s| s.as_slice()).collect();
    let r = PhraseRepeats::detect(&refs);
    // A single shared position cannot satisfy both slots.
    assert!(!sloppy_phrase_matches(&[&[4][..], &[4][..]], &r, 50));
    // Two positions can.
    assert!(sloppy_phrase_matches(&[&[4, 5][..], &[4, 5][..]], &r, 1));
}

#[test]
fn a_collision_created_mid_walk_is_resolved_and_the_queue_re_ordered() {
    // The `advanceRpts` path: with three occurrences of one term and a
    // two-slot repeat group, advancing the leading slot lands it on the
    // trailing slot's raw position, which Java resolves by advancing the
    // *lesser* of the two and re-queuing whatever moved. Nothing else in
    // this module's tests reaches it, because it needs a repeat group whose
    // members can still collide after initialization spread them apart.
    let rpt = repeats_for(&["a", "a"]);
    let three = [&[0i32, 1, 2][..], &[0i32, 1, 2][..]];
    assert!(sloppy_phrase_matches(&three, &rpt, 0));
    // At slop 0 the answer must be the exact matcher's: "a a" occurs at
    // (0,1) and (1,2), so twice, each weighing 1.
    assert_eq!(sloppy_phrase_freq(&three, &rpt, 0), 2.0);
    // Four occurrences, three matches.
    let four = [&[0i32, 1, 2, 3][..], &[0i32, 1, 2, 3][..]];
    assert_eq!(sloppy_phrase_freq(&four, &rpt, 0), 3.0);
    // A gap the group has to step over: occurrences at 0 and 5 are four
    // apart, so `"a a"` needs slop 4 and weighs 1/5.
    let apart = [&[0i32, 5][..], &[0i32, 5][..]];
    assert!(!sloppy_phrase_matches(&apart, &rpt, 3));
    assert!(sloppy_phrase_matches(&apart, &rpt, 4));
    assert_eq!(sloppy_phrase_freq(&apart, &rpt, 4), 1.0 / 5.0);
}

#[test]
fn the_walk_shrinks_a_match_before_re_queuing() {
    // `nextMatch`'s "keep advancing the same pp while it is still below the
    // queue's next element" refinement: the first window found for a start
    // is not necessarily the tightest, and Lucene reports the tightest.
    // a@[0, 9], b@[7]: shifted to a=[0, 9], b=[6]. Popping a@0 gives
    // `end - 0 == 6`; advancing a to 9 is still not past the queue's next
    // element's position, and shrinks the window to `9 - 6 == 3`.
    let got = freq(&[&[0i32, 9][..], &[7i32][..]], 3);
    assert_eq!(got, 1.0 / 4.0, "the reported match length must be 3, not 6");
    assert!(!matches(&[&[0i32, 9][..], &[7i32][..]], 2));
}

#[test]
fn a_repeat_group_that_runs_out_of_positions_does_not_match() {
    let rpt = repeats_for(&["a", "a", "a"]);
    // Only two occurrences for three slots.
    assert!(!sloppy_phrase_matches(
        &[&[0, 1][..], &[0, 1][..], &[0, 1][..]],
        &rpt,
        100
    ));
    assert_eq!(
        sloppy_phrase_freq(&[&[0, 1][..], &[0, 1][..], &[0, 1][..]], &rpt, 100),
        0.0
    );
}

#[test]
fn degenerate_shapes_match_nothing_or_the_term_frequency() {
    assert!(!matches(&[], 5));
    assert!(!matches(&[&[][..]], 5));
    assert!(!matches(&[&[0][..], &[][..]], 5));
    assert_eq!(freq(&[], 5), 0.0);
    assert_eq!(freq(&[&[][..], &[1][..]], 5), 0.0);
    // A one-slot phrase degenerates to the term's own frequency.
    assert!(matches(&[&[2, 9][..]], 0));
    assert_eq!(freq(&[&[2, 9][..]], 0), 2.0);
}

#[test]
fn two_slots_walk_to_the_general_matchers_frequency_bit_for_bit() {
    // Every shape of two short position lists (both slots distinct terms):
    // the two-pointer walk against the general matcher.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rand = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    let list = |rand: &mut dyn FnMut(u64) -> u64| {
        let len = 1 + rand(6) as usize;
        let mut v: Vec<i32> = (0..len).map(|_| rand(20) as i32).collect();
        v.sort_unstable();
        v
    };
    for _ in 0..20_000 {
        let a = list(&mut rand);
        let b = list(&mut rand);
        let slop = rand(6) as u32;
        let lists = [&a[..], &b[..]];
        let general = {
            let mut m = SloppyMatcher::new(&lists, &PhraseRepeats::none(2), slop);
            let mut f = 0.0f32;
            if m.next_match() {
                f = m.sloppy_weight();
                while m.next_match() {
                    f += m.sloppy_weight();
                }
            }
            f
        };
        let fast = two_slot_freq(&a, &b, slop);
        assert_eq!(fast.to_bits(), general.to_bits(), "{a:?} {b:?} slop {slop}");
        // The first match alone agrees on whether there is one, for two
        // slots and for three, with and without a repeated term.
        let mut scratch = SloppyScratch::default();
        let none2 = PhraseRepeats::none(2);
        let first = sloppy_phrase_matches_in(&mut scratch, &lists, &none2, slop, None);
        assert_eq!(first, general > 0.0, "{a:?} {b:?} slop {slop}");
        let c = list(&mut rand);
        let three = [&a[..], &b[..], &c[..]];
        for repeats in [
            PhraseRepeats::none(3),
            PhraseRepeats::for_phrase(&[b"x".to_vec(), b"y".to_vec(), b"x".to_vec()]),
        ] {
            let freq = sloppy_phrase_freq(&three, &repeats, slop);
            let first = sloppy_phrase_matches_in(&mut scratch, &three, &repeats, slop, None);
            assert_eq!(first, freq > 0.0, "{a:?} {b:?} {c:?} slop {slop}");
        }
    }
    assert_eq!(two_slot_freq(&[], &[1], 3), 0.0);
}

#[test]
fn several_occurrences_sum_their_weights() {
    // "a b" over a@[0,4] b@[1,7]: an adjacent pair (weight 1) and a pair
    // three apart (weight 1/3).
    let got = freq(&[&[0, 4][..], &[1, 7][..]], 3);
    assert_eq!(got, 1.0 + 1.0 / 3.0);
}

#[test]
fn a_match_beyond_the_budget_contributes_nothing() {
    // a@0 b@6: window 5, over a budget of 2.
    assert_eq!(freq(&[&[0][..], &[6][..]], 2), 0.0);
}

#[test]
fn slop_zero_agrees_with_exact_adjacency_on_every_small_case() {
    // Cross-check against the exact matcher's own rule for every pair of
    // three-position lists, which is the invariant that lets
    // `search_phrase_query` keep using the exact fast path at slop 0.
    for a in 0..8i32 {
        for b in 0..8i32 {
            let exact = b == a + 1;
            assert_eq!(matches(&[&[a][..], &[b][..]], 0), exact, "a={a} b={b}");
        }
    }
}
