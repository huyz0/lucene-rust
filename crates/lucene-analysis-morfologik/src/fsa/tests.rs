#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk
use super::*;

/// A CFSA2 holding "ab" and "ac" (explicit labels, no mapping).
fn cfsa2() -> Vec<u8> {
    // Node 0 (epsilon): one arc, label 0, last, target = 3 (v-int).
    // Node 3 (root): 'a' last, target 7.
    // Node 7: 'b' final, terminal (target 0); 'c' final last, terminal.
    let mut f = b"\\fsa\xC6".to_vec();
    f.extend_from_slice(&[0, 0, 0]); // flags, empty label mapping
    let arcs = [
        0x40, 0x00, 0x03, // 0: epsilon arc -> 3
        0x40, b'a', 0x07, // 3: 'a' -> 7
        0x00, // pad (6)
        0x20, b'b', 0x00, // 7: 'b' final, terminal
        0x60, b'c', 0x00, // 10: 'c' final last, terminal
    ];
    f.extend_from_slice(&arcs);
    f
}

#[test]
fn reads_and_walks_a_cfsa2() {
    let fsa = Fsa::read(&cfsa2()).unwrap();
    let root = fsa.root().unwrap();
    assert_eq!(root, 3);
    assert_eq!(fsa.match_sequence(b"ab", root).unwrap(), Match::Exact);
    assert_eq!(
        fsa.match_sequence(b"a", root).unwrap(),
        Match::SequenceIsAPrefix(7)
    );
    assert_eq!(
        fsa.match_sequence(b"abc", root).unwrap(),
        Match::AutomatonHasPrefix
    );
    assert_eq!(
        fsa.match_sequence(b"ax", root).unwrap(),
        Match::AutomatonHasPrefix
    );
    assert_eq!(fsa.match_sequence(b"x", root).unwrap(), Match::No);
    assert_eq!(fsa.match_sequence(b"x", 0).unwrap(), Match::No);
    assert_eq!(
        fsa.sequences(root).unwrap(),
        vec![b"ab".to_vec(), b"ac".to_vec()]
    );
    assert_eq!(
        fsa.sequences(7).unwrap(),
        vec![b"b".to_vec(), b"c".to_vec()]
    );
}

#[test]
fn refuses_bad_headers() {
    for bad in [
        &b"\\fs"[..],
        b"xfsa\x05",
        b"\\fsa",
        b"\\fsa\xC5",
        b"\\fsa\x05\x00",
        b"\\fsa\xC6\x00",
        b"\\fsa\xC6\x00\x00\x05ab",
        b"\\fsa\xC6\x40\x00\x00",
    ] {
        assert!(Fsa::read(bad).is_err(), "{bad:?}");
    }
    assert!(Fsa::read(b"\\fsa\xC6\x01\x00\x00").is_ok());
}

#[test]
fn corrupt_offsets_are_errors() {
    let mut f = cfsa2();
    let len = f.len();
    f[len - 4] = 0x50; // 'b' targets a node past the arcs
    let fsa = Fsa::read(&f).unwrap();
    assert!(fsa.sequences(7).is_err());
    let mut f = cfsa2();
    f[len - 6] = 0x21; // 'b': a label index past the (empty) mapping
    let fsa = Fsa::read(&f).unwrap();
    assert!(fsa.sequences(7).is_err());
    // A truncated automaton.
    let fsa = Fsa::read(&cfsa2()[..12]).unwrap();
    assert!(fsa.root().is_err() || fsa.sequences(3).is_err());
    // A cycle: the 'a' arc points back at its own node.
    let mut c = cfsa2();
    c[8 + 5] = 0x03;
    let fsa = Fsa::read(&c).unwrap();
    assert!(fsa.sequences(3).is_err());
}

/// A CFSA2 whose root reads `a`, then `+`, then the node at 9: `tail`.
fn after_separator(tail: &[u8]) -> Vec<u8> {
    let mut f = b"\\fsa\xC6\x00\x00\x00".to_vec();
    f.extend_from_slice(&[0x40, b'x', 3, 0x40, b'a', 6, 0x40, b'+', 9]);
    f.extend_from_slice(tail);
    f
}

#[test]
fn a_final_cycle_stops_at_the_automatons_size() {
    // 9: 'b', final and last, back to 9 -- every "b...b" is a sequence.
    // Bounded only by the 65,536-level cap, the walk held 2 GB of them
    // before it stopped.
    let fsa = Fsa::read(&after_separator(&[0x60, b'b', 9])).unwrap();
    let e = fsa.sequences(9).unwrap_err();
    assert!(e.message().contains("cycle"), "{}", e.message());
}

#[test]
fn a_lookups_sequences_are_capped() {
    // 17 levels of two final arcs each: 2^18 - 2 sequences from 102
    // bytes of arcs.
    let mut dag = Vec::new();
    for k in 0..17u8 {
        let next = if k == 16 { 0 } else { 9 + 6 * (k + 1) };
        dag.extend_from_slice(&[0x20, b'b', next, 0x60, b'c', next]);
    }
    let fsa = Fsa::read(&after_separator(&dag)).unwrap();
    let e = fsa.sequences(9).unwrap_err();
    assert!(e.message().contains("65536 sequences"), "{}", e.message());
    // A chain of n final arcs: n sequences, n(n+1)/2 bytes.
    let chain = |n: usize| {
        let mut c = Vec::new();
        for _ in 1..n {
            c.extend_from_slice(&[0xE0, b'b']); // final, last, target next
        }
        c.extend_from_slice(&[0x60, b'b', 0]);
        Fsa::read(&after_separator(&c)).unwrap()
    };
    let e = chain(6000).sequences(9).unwrap_err();
    assert!(e.message().contains("bytes"), "{}", e.message());
    let ok = chain(100).sequences(9).unwrap();
    assert_eq!(ok.len(), 100);
    assert_eq!(ok[99], vec![b'b'; 100]);
}

#[test]
fn target_next_and_mapped_labels() {
    // Mapping [_, 'q']. Node 3: 'b' (final, target next, not last),
    // 'c' (final, last, terminal); node 8: mapped 'q' (final, last,
    // terminal).
    let mut f = b"\\fsa\xC6\x00\x00\x02\x00q".to_vec();
    f.extend_from_slice(&[0x40, 0x00, 0x03, 0xA0, b'b', 0x60, b'c', 0x00, 0x61, 0x00]);
    let fsa = Fsa::read(&f).unwrap();
    assert_eq!(fsa.root().unwrap(), 3);
    assert_eq!(
        fsa.sequences(3).unwrap(),
        vec![b"b".to_vec(), b"bq".to_vec(), b"c".to_vec()]
    );
    assert_eq!(fsa.arc(3, b'c').unwrap(), 5);
    assert_eq!(fsa.arc(3, b'x').unwrap(), 0);
    assert_eq!(fsa.arc(8, b'q').unwrap(), 8);
    assert_eq!(fsa.label(8).unwrap(), b'q');
    assert_eq!(fsa.next_arc(3).unwrap(), 5);
    assert_eq!(fsa.next_arc(5).unwrap(), 0);
    assert!(fsa.is_final(3).unwrap());
    assert_eq!(fsa.end_node(3).unwrap(), 8);
    assert_eq!(fsa.end_node(5).unwrap(), 0);
    assert_eq!(fsa.match_sequence(b"bq", 3).unwrap(), Match::Exact);
    // The same scans over a truncated copy fail cleanly.
    let fsa = Fsa::read(&f[..f.len() - 3]).unwrap();
    assert!(fsa.sequences(3).is_err());
    assert!(fsa.arc(8, b'x').is_err());
}

#[test]
fn fsa5_arcs_one_by_one() {
    let arc = |label: u8, target: u16, flags: u16| {
        let v = (target << 3) | flags;
        [label, (v & 0xFF) as u8, (v >> 8) as u8]
    };
    // Root 6: 'a' (not last) -> 12, 'b' (final, last, terminal);
    // 12: 'c' (final, last, terminal).
    let mut f = b"\\fsa\x05_+\x02".to_vec();
    f.extend_from_slice(&arc(0, 0, 2));
    f.extend_from_slice(&arc(0, 6, 2));
    f.extend_from_slice(&arc(b'a', 12, 0));
    f.extend_from_slice(&arc(b'b', 0, 3));
    f.extend_from_slice(&arc(b'c', 0, 3));
    let fsa = Fsa::from_vec(f.clone()).unwrap();
    assert_eq!(fsa.arc(6, b'b').unwrap(), 9);
    assert_eq!(fsa.arc(6, b'x').unwrap(), 0);
    assert_eq!(fsa.label(9).unwrap(), b'b');
    assert_eq!(fsa.next_arc(6).unwrap(), 9);
    assert_eq!(fsa.next_arc(9).unwrap(), 0);
    assert!(!fsa.is_final(6).unwrap());
    assert_eq!(fsa.end_node(6).unwrap(), 12);
    assert_eq!(
        fsa.sequences(6).unwrap(),
        vec![b"ac".to_vec(), b"b".to_vec()]
    );
    let fsa = Fsa::read(&f[..f.len() - 2]).unwrap();
    assert!(fsa.sequences(6).is_err());
    assert!(fsa.arc(12, b'x').is_err());
}

#[test]
fn vints_wrap_as_java_ints() {
    let c = Cfsa2 {
        arcs: vec![0xFF, 0xFF, 0xFF, 0xFF, 0x7F],
        label_mapping: vec![],
        has_numbers: true,
    };
    assert!(c.read_vint(0).is_err()); // negative as an int
    let c = Cfsa2 {
        arcs: vec![0x81, 0x01, 0x02],
        label_mapping: vec![],
        has_numbers: true,
    };
    assert_eq!(c.read_vint(0).unwrap(), 129);
    assert_eq!(c.first_arc(0).unwrap(), 2);
}

/// An FSA5 with `gtl` 2 holding "ab": a dummy arc at 0, the epsilon
/// node at 3 pointing at the root, 6.
#[test]
fn reads_and_walks_an_fsa5() {
    let arc = |label: u8, target: u16, flags: u16| {
        let v = (target << 3) | flags;
        [label, (v & 0xFF) as u8, (v >> 8) as u8]
    };
    let mut f = b"\\fsa\x05_+\x02".to_vec();
    f.extend_from_slice(&arc(0, 0, 2));
    f.extend_from_slice(&arc(0, 6, 2));
    f.extend_from_slice(&arc(b'a', 9, 2));
    f.extend_from_slice(&arc(b'b', 0, 3));
    let fsa = Fsa::read(&f).unwrap();
    let root = fsa.root().unwrap();
    assert_eq!(root, 6);
    assert_eq!(fsa.match_sequence(b"ab", root).unwrap(), Match::Exact);
    assert_eq!(fsa.sequences(root).unwrap(), vec![b"ab".to_vec()]);
    // A "next" arc: the target follows it.
    let mut n = b"\\fsa\x05_+\x02".to_vec();
    n.extend_from_slice(&arc(0, 0, 2));
    n.extend_from_slice(&arc(0, 6, 2));
    n.extend_from_slice(&[b'a', 0x06]); // 6: 'a', last + next -> 8
    n.extend_from_slice(&arc(b'b', 0, 3)); // 8
    let fsa = Fsa::read(&n).unwrap();
    let root = fsa.root().unwrap();
    assert_eq!(fsa.sequences(root).unwrap(), vec![b"ab".to_vec()]);
    assert!(Fsa::read(b"\\fsa\x05_").is_err());
}
