#![cfg(test)]
use super::*;

fn terms(v: &[&str]) -> Vec<Vec<u8>> {
    v.iter().map(|s| s.as_bytes().to_vec()).collect()
}

fn cursors<'a>(segments: &'a [Vec<Vec<u8>>]) -> Vec<SliceCursor<'a, Vec<u8>>> {
    segments
        .iter()
        .map(|terms| SliceCursor { terms, next: 0 })
        .collect()
}

/// A map remapped over segments of its own and others whose terms it
/// holds, in any order, is the map built over them; a new term, or a
/// term left in no segment, leaves it to be built.
#[test]
fn remapped_maps_equal_built_ones() {
    let base = vec![terms(&["a", "c", "e"]), terms(&["b", "c"])];
    let mut cs = cursors(&base);
    let mut refs: Vec<&mut dyn TermCursor> = cs.iter_mut().map(|c| c as _).collect();
    let (map, kept) = OrdinalMap::build_keeping_terms(&mut refs, 1 << 10).unwrap();
    let kept = kept.unwrap();
    assert_eq!(kept.len(), 4);
    let remap = |order: &[Option<usize>], more: &[Vec<Vec<u8>>]| {
        let mut cs = cursors(more);
        let mut more_cs = cs.iter_mut();
        let mut sources: Vec<SegmentSource<'_>> = order
            .iter()
            .map(|o| match o {
                Some(i) => SegmentSource::Kept(*i),
                None => SegmentSource::Terms(more_cs.next().unwrap()),
            })
            .collect();
        map.remapped(&kept, &mut sources).unwrap()
    };
    let more = vec![terms(&["c", "e"]), terms(&[]), terms(&["a", "b", "e"])];
    // Appended after the map's segments.
    let got = remap(&[Some(0), Some(1), None, None, None], &more).unwrap();
    let all: Vec<Vec<Vec<u8>>> = base.iter().chain(&more).cloned().collect();
    assert_eq!(got, OrdinalMap::build(&all));
    // Interleaved and reordered.
    let got = remap(&[Some(1), None, Some(0), None, None], &more).unwrap();
    let all = vec![
        base[1].clone(),
        more[0].clone(),
        base[0].clone(),
        more[1].clone(),
        more[2].clone(),
    ];
    assert_eq!(got, OrdinalMap::build(&all));
    // A segment dropped, its terms still held by the others.
    let got = remap(&[None, Some(1), None, None], &more).unwrap();
    let all = vec![
        more[0].clone(),
        base[1].clone(),
        more[1].clone(),
        more[2].clone(),
    ];
    assert_eq!(got, OrdinalMap::build(&all));
    // A term in no segment any more.
    assert!(remap(&[Some(1)], &[]).is_none());
    // A new term.
    assert!(remap(&[Some(0), None], &[terms(&["c", "d"])]).is_none());
    // Four terms of four bytes together are at the limit of four.
    let mut cs = cursors(&base);
    let mut refs: Vec<&mut dyn TermCursor> = cs.iter_mut().map(|c| c as _).collect();
    let at_limit = OrdinalMap::build_keeping_terms(&mut refs, 4).unwrap().1;
    assert_eq!(at_limit.map(|t| t.len()), Some(4));
    // Terms past the limit are not kept.
    let mut cs = cursors(&base);
    let mut refs: Vec<&mut dyn TermCursor> = cs.iter_mut().map(|c| c as _).collect();
    assert!(OrdinalMap::build_keeping_terms(&mut refs, 3)
        .unwrap()
        .1
        .is_none());
}

/// The exact worked example in `OrdinalMap`'s own class javadoc:
/// global `bar -> 0, cat -> 1, dog -> 2, foo -> 3`; segment 0 holds
/// `bar, foo`, segment 1 holds `cat, dog`.
#[test]
fn a_map_counts_its_bytes_while_it_lives() {
    let map = OrdinalMap::build(&[terms(&["bar", "foo"]), terms(&["cat", "dog"])]);
    // Two segments of two ordinals (8 bytes each), four first segments
    // (4) and their ordinals (8).
    assert_eq!(map.held.bytes(), 4 * 8 + 4 * 4 + 4 * 8);
    assert!(held_bytes() >= map.held.bytes());
    let clone = map.clone();
    assert!(
        Arc::ptr_eq(&map.held, &clone.held),
        "a clone is not counted again"
    );
}

#[test]
fn javadoc_worked_example() {
    let map = OrdinalMap::build(&[terms(&["bar", "foo"]), terms(&["cat", "dog"])]);
    assert_eq!(map.value_count(), 4);
    assert_eq!(map.segment_ords(0), Some(&[0i64, 3][..]));
    assert_eq!(map.segment_ords(1), Some(&[1i64, 2][..]));
}

#[test]
fn a_term_shared_by_every_segment_gets_one_global_ordinal() {
    let map = OrdinalMap::build(&[
        terms(&["a", "b"]),
        terms(&["a", "c"]),
        terms(&["a", "b", "c"]),
    ]);
    assert_eq!(map.value_count(), 3);
    assert_eq!(map.segment_ords(0), Some(&[0i64, 1][..]));
    assert_eq!(map.segment_ords(1), Some(&[0i64, 2][..]));
    assert_eq!(map.segment_ords(2), Some(&[0i64, 1, 2][..]));
    // "first segment" ties break to the lowest input index.
    assert_eq!(map.first_segment(0), Some(0));
    assert_eq!(map.first_segment(2), Some(1));
    assert_eq!(map.first_segment_ord(2), Some(1));
}

#[test]
fn an_empty_segment_contributes_nothing_and_is_still_addressable() {
    let map = OrdinalMap::build(&[terms(&["a"]), Vec::<Vec<u8>>::new(), terms(&["b"])]);
    assert_eq!(map.value_count(), 2);
    assert_eq!(map.segment_count(), 3);
    assert_eq!(map.segment_ords(1), Some(&[][..]));
    assert_eq!(map.global_ord(1, 0), None);
    assert_eq!(map.global_ord(2, 0), Some(1));
}

#[test]
fn no_segments_at_all_is_an_empty_map_not_a_panic() {
    let map = OrdinalMap::build::<Vec<u8>>(&[]);
    assert_eq!(map.value_count(), 0);
    assert_eq!(map.segment_count(), 0);
    assert_eq!(map.global_ord(0, 0), None);
    assert_eq!(map.first_segment(0), None);
    assert_eq!(map.first_segment_ord(0), None);
}

#[test]
fn out_of_range_lookups_return_none_rather_than_panicking() {
    let map = OrdinalMap::build(&[terms(&["a", "b"])]);
    assert_eq!(map.global_ord(0, 1), Some(1));
    assert_eq!(map.global_ord(0, 2), None);
    assert_eq!(map.global_ord(0, -1), None);
    assert_eq!(map.global_ord(9, 0), None);
    assert_eq!(map.segment_ords(9), None);
    assert_eq!(map.first_segment(-1), None);
    assert_eq!(map.first_segment_ord(-1), None);
    assert_eq!(map.first_segment(99), None);
    assert_eq!(map.first_segment_ord(99), None);
}

/// The global order is byte order, not the order segments were merged in:
/// a term that sorts first but appears only in the *last* segment still
/// takes global ordinal 0.
#[test]
fn global_ordinals_follow_byte_order_across_segments() {
    let map = OrdinalMap::build(&[terms(&["m", "z"]), terms(&["b", "m"])]);
    assert_eq!(map.segment_ords(0), Some(&[1i64, 2][..]));
    assert_eq!(map.segment_ords(1), Some(&[0i64, 1][..]));
    assert_eq!(map.first_segment(0), Some(1));
}

/// Byte order, not `str` order: a term with a byte above 0x7F must not be
/// compared as UTF-8 code points (Java compares `BytesRef`s).
#[test]
fn ordering_is_over_raw_bytes() {
    let a = vec![vec![0xC3u8, 0xA9]]; // "é"
    let b = vec![vec![0x7Au8]]; // "z"
    let map = OrdinalMap::build(&[a, b]);
    assert_eq!(map.segment_ords(1), Some(&[0i64][..]), "0x7A < 0xC3");
    assert_eq!(map.segment_ords(0), Some(&[1i64][..]));
}

/// A [`TermCursor`] over an owned list, so the streaming entry point can
/// be driven from a test without a `.dvd`. Deliberately hands back a
/// **reused** buffer, which is the contract a real
/// `terms_dict::TermsCursor` relies on: a merge that assumed the previous
/// term stayed valid would read the wrong bytes here.
struct VecCursor {
    terms: Vec<Vec<u8>>,
    next: usize,
    buffer: Vec<u8>,
    fail_at: Option<usize>,
}

impl VecCursor {
    fn new(v: &[&str]) -> Self {
        VecCursor {
            terms: terms(v),
            next: 0,
            buffer: Vec::new(),
            fail_at: None,
        }
    }
}

impl TermCursor for VecCursor {
    fn next_term(&mut self) -> lucene_store::Result<Option<&[u8]>> {
        if self.fail_at == Some(self.next) {
            return Err(lucene_store::Error::Corrupted("bad block".into()));
        }
        let Some(term) = self.terms.get(self.next) else {
            return Ok(None);
        };
        self.next += 1;
        self.buffer.clear();
        self.buffer.extend_from_slice(term);
        Ok(Some(&self.buffer))
    }
}

fn streamed(mut cursors: Vec<VecCursor>) -> lucene_store::Result<OrdinalMap> {
    let mut refs: Vec<&mut dyn TermCursor> = cursors
        .iter_mut()
        .map(|c| c as &mut dyn TermCursor)
        .collect();
    OrdinalMap::build_streaming(&mut refs)
}

/// The two entry points are one algorithm, so they must agree exactly --
/// on every field of the map, not just `valueCount`. Eight segments, so
/// the heap is deep enough that both children of a node are compared on
/// the way down (a two- or three-segment case never reaches the right
/// child).
#[test]
fn build_streaming_agrees_with_build_over_the_same_terms() {
    let shapes: [&[&[&str]]; 4] = [
        &[&["bar", "foo"], &["cat", "dog"]],
        &[&["a", "b"], &["a", "c"], &["a", "b", "c"]],
        &[&["a"], &[], &["b"]],
        &[
            &["h", "p"],
            &["a", "z"],
            &["c", "h", "q"],
            &["b"],
            &["d", "e", "f"],
            &["g", "h"],
            &["m"],
            &["a", "n", "z"],
        ],
    ];
    for shape in shapes {
        let materialized: Vec<Vec<Vec<u8>>> = shape.iter().map(|s| terms(s)).collect();
        let expected = OrdinalMap::build(&materialized);
        let got = streamed(shape.iter().map(|s| VecCursor::new(s)).collect()).unwrap();
        assert_eq!(got, expected, "shape {shape:?}");
    }
}

/// The eight-segment shape spelled out, so the agreement test above
/// cannot pass by both halves being wrong in the same way.
#[test]
fn build_streaming_assigns_global_ordinals_in_byte_order() {
    let map = streamed(vec![
        VecCursor::new(&["c", "e"]),
        VecCursor::new(&["a", "e"]),
        VecCursor::new(&["b", "d"]),
    ])
    .unwrap();
    // Global: a=0, b=1, c=2, d=3, e=4.
    assert_eq!(map.value_count(), 5);
    assert_eq!(map.segment_ords(0), Some(&[2i64, 4][..]));
    assert_eq!(map.segment_ords(1), Some(&[0i64, 4][..]));
    assert_eq!(map.segment_ords(2), Some(&[1i64, 3][..]));
    // "e" is in segments 0 and 1; "first" is the lowest input index, and
    // its ordinal there is 1.
    assert_eq!(map.first_segment(4), Some(0));
    assert_eq!(map.first_segment_ord(4), Some(1));
}

/// A cursor's error is the caller's, not a panic and not a silently short
/// map -- the failure mode a corrupt `.dvd` produces.
#[test]
fn a_cursor_error_propagates_out_of_build_streaming() {
    let mut ok = VecCursor::new(&["a", "b"]);
    let mut bad = VecCursor::new(&["a", "c"]);
    bad.fail_at = Some(1);
    let mut refs: Vec<&mut dyn TermCursor> = vec![
        &mut ok as &mut dyn TermCursor,
        &mut bad as &mut dyn TermCursor,
    ];
    assert!(matches!(
        OrdinalMap::build_streaming(&mut refs),
        Err(lucene_store::Error::Corrupted(_))
    ));
}

/// The very first `next_term` failing is a separate branch from one
/// failing mid-merge (the priming loop, not the drain loop).
#[test]
fn a_cursor_that_fails_on_its_first_term_propagates_too() {
    let mut bad = VecCursor::new(&["a"]);
    bad.fail_at = Some(0);
    let mut refs: Vec<&mut dyn TermCursor> = vec![&mut bad as &mut dyn TermCursor];
    assert!(matches!(
        OrdinalMap::build_streaming(&mut refs),
        Err(lucene_store::Error::Corrupted(_))
    ));
}

#[test]
fn build_streaming_over_no_cursors_is_an_empty_map() {
    let map = streamed(Vec::new()).unwrap();
    assert_eq!(map.value_count(), 0);
    assert_eq!(map.segment_count(), 0);
}

/// The streaming entry point carries the same debug-time guard as the
/// materialized one, and it has to be its own check: `build`'s runs over
/// the lists before the merge starts, and a cursor has no list to scan.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "not strictly ascending")]
fn an_unsorted_cursor_is_caught_in_debug_builds() {
    let _ = streamed(vec![VecCursor::new(&["b", "a"])]);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "not strictly ascending")]
fn unsorted_input_is_caught_in_debug_builds() {
    OrdinalMap::build(&[terms(&["b", "a"])]);
}
