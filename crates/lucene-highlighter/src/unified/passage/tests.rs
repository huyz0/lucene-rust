use super::*;

fn u16s(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

fn passage(start: i32, end: i32, matches: &[(i32, i32)]) -> Passage {
    let mut p = Passage::new();
    p.set_start_offset(start);
    p.set_end_offset(end);
    for &(s, e) in matches {
        p.add_match(s, e, b"t", 1);
    }
    p
}

#[test]
fn a_passage_reports_and_resets_what_was_added() {
    let mut p = passage(4, 40, &[(5, 8), (10, 12)]);
    p.set_score(2.5);
    assert_eq!((p.start_offset(), p.end_offset(), p.length()), (4, 40, 36));
    assert_eq!(p.num_matches(), 2);
    assert_eq!(p.match_starts(), &[5, 10]);
    assert_eq!(p.match_ends(), &[8, 12]);
    assert_eq!(p.match_terms(), &[b"t".to_vec(), b"t".to_vec()]);
    assert_eq!(p.match_term_freqs_in_doc(), &[1, 1]);
    assert_eq!(p.to_string(), "Passage[4-40]{t[1-4],t[6-8]}score=2.5");
    p.reset();
    assert_eq!(
        (p.start_offset(), p.end_offset(), p.num_matches()),
        (-1, -1, 0)
    );
    assert_eq!(p.score(), 0.0);
    assert_eq!(Passage::default().to_string(), "Passage[-1--1]{}score=0");
}

#[test]
fn the_formatter_escapes_the_owasp_characters_and_merges_overlaps() {
    let content = u16s("a<b>&\"'/ cd efgh");
    let f = DefaultPassageFormatter::new("[", "]", "~", true);
    let p = passage(0, 16, &[(0, 3), (2, 5), (9, 11)]);
    assert_eq!(
        f.format(&[p], &content),
        "[a&lt;b&gt;&amp;]&quot;&#x27;&#x2F; [cd] efgh"
    );
}

#[test]
fn disjoint_passages_are_joined_by_the_ellipsis_and_adjacent_ones_are_not() {
    let content = u16s("one two three four five six");
    let f = DefaultPassageFormatter::default();
    let a = passage(0, 8, &[(0, 3)]);
    let b = passage(8, 14, &[(8, 13)]);
    assert_eq!(
        f.format(&[a.clone(), b.clone()], &content),
        "<b>one</b> two <b>three</b> "
    );
    let far = passage(14, 19, &[(14, 18)]);
    assert_eq!(
        f.format(&[a, far], &content),
        "<b>one</b> two ... <b>four</b> "
    );
    // a match straddling the passage end is cut to it
    let d = passage(0, 5, &[(2, 9)]);
    assert_eq!(f.format(&[d], &content), "on<b>e t</b>");
}

#[test]
fn the_scorer_follows_bm25() {
    let s = PassageScorer::new(1.2, 0.75, 87.0);
    assert!(s.weight(1000, 3) > s.weight(1000, 300));
    assert!(s.tf(4, 80) > s.tf(1, 80));
    assert!(s.norm(0) > s.norm(5000));
    let mut p = passage(0, 50, &[(0, 3), (10, 13)]);
    assert!(s.score(&p, 1000) > 0.0);
    p.reset();
    p.set_start_offset(0);
    p.set_end_offset(50);
    assert_eq!(s.score(&p, 1000), 0.0);
}

#[test]
#[should_panic(expected = "UnsupportedOperationException")]
fn the_empty_enum_has_no_term() {
    let _ = EmptyOffsets.term();
}

#[test]
#[should_panic(expected = "UnsupportedOperationException")]
fn the_empty_enum_has_no_start() {
    let _ = EmptyOffsets.start_offset();
}

#[test]
#[should_panic(expected = "UnsupportedOperationException")]
fn the_empty_enum_has_no_end() {
    let _ = EmptyOffsets.end_offset();
}

#[test]
fn enums_compare_by_offsets_then_term() {
    let mut empty = EmptyOffsets;
    assert!(!empty.next_position().unwrap());
    assert_eq!(empty.freq(), 0);
    let mut a = OfPostings::new(b"a".to_vec(), 1, vec![(1, 2)]);
    let mut b = OfPostings::new(b"b".to_vec(), 1, vec![(1, 2)]);
    let mut c = OfPostings::new(b"a".to_vec(), 1, vec![(1, 3)]);
    let mut d = OfPostings::new(b"a".to_vec(), 1, vec![(0, 3)]);
    for e in [&mut a, &mut b, &mut c, &mut d] {
        assert!(e.next_position().unwrap());
    }
    assert_eq!(compare_offsets(&a, &b), Ordering::Less);
    assert_eq!(compare_offsets(&a, &c), Ordering::Less);
    assert_eq!(compare_offsets(&a, &d), Ordering::Greater);
    assert_eq!(compare_offsets(&a, &a), Ordering::Equal);
    assert!(!a.next_position().unwrap());
}

/// An enum with no term, like a wildcard automaton's.
struct Anonymous;

impl OffsetsEnum for Anonymous {
    fn next_position(&mut self) -> Result<bool> {
        Ok(true)
    }
    fn freq(&self) -> i32 {
        1
    }
    fn term(&self) -> Option<&[u8]> {
        None
    }
    fn start_offset(&self) -> i32 {
        1
    }
    fn end_offset(&self) -> i32 {
        2
    }
}

#[test]
fn a_missing_term_sorts_after_a_present_one() {
    let mut a = OfPostings::new(b"a".to_vec(), 1, vec![(1, 2)]);
    a.next_position().unwrap();
    assert_eq!(compare_offsets(&Anonymous, &a), Ordering::Greater);
    assert_eq!(compare_offsets(&a, &Anonymous), Ordering::Less);
    assert_eq!(compare_offsets(&Anonymous, &Anonymous), Ordering::Equal);
}

#[test]
fn a_merge_walks_its_parts_in_offset_order() {
    let parts: Vec<Box<dyn OffsetsEnum>> = vec![
        Box::new(OfPostings::new(b"x".to_vec(), 2, vec![(5, 6), (20, 21)])),
        Box::new(OfMatches::new(vec![
            MatchEntry {
                term: b"y".to_vec(),
                start: 1,
                end: 2,
                freq: 7,
            },
            MatchEntry {
                term: b"y".to_vec(),
                start: 9,
                end: 10,
                freq: 7,
            },
        ])),
        Box::new(OfPostings::new(b"z".to_vec(), 0, vec![])),
    ];
    let mut m = MultiOffsets::new(parts).unwrap();
    let mut seen = Vec::new();
    while m.next_position().unwrap() {
        seen.push((m.start_offset(), m.end_offset(), m.freq()));
    }
    let starts: Vec<i32> = seen.iter().map(|s| s.0).collect();
    assert_eq!(starts, [1, 5, 9, 20]);
    assert_eq!(seen[0].2, 7);
    assert!(!m.next_position().unwrap());
    let mut none = MultiOffsets::new(Vec::new()).unwrap();
    assert!(!none.next_position().unwrap());
}
