#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn dfa_of(build: impl FnOnce(&mut Nfa, u32, u32)) -> ByteDfa {
    let mut nfa = Nfa::new();
    let s = nfa.state().unwrap();
    let a = nfa.state().unwrap();
    build(&mut nfa, s, a);
    nfa.determinize(s, a).expect("determinize")
}

#[test]
fn utf8_sequences_cover_exactly_the_code_point_range() {
    for (lo, hi) in [
        (0u32, 0x7F),
        (0x41, 0x5A),
        (0x7E, 0x81),
        (0x7FF, 0x800),
        (0x100, 0x2000),
        (0xD7FF, 0xE001),
        (0xFFFF, 0x10000),
        (0, MAX_CODE_POINT),
        (0x10FFFE, MAX_CODE_POINT),
    ] {
        let dfa = dfa_of(|nfa, s, a| {
            nfa.code_points(s, lo, hi, a).unwrap();
        });
        // Sample the whole space densely at the edges and sparsely inside.
        let probes = (0..0x900u32)
            .chain((0xD700..0xE100).step_by(7))
            .chain((0xF000..0x10100).step_by(13))
            .chain((0x10F000..=MAX_CODE_POINT).step_by(3));
        for cp in probes {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let mut buf = [0u8; 4];
            let bytes = c.encode_utf8(&mut buf).as_bytes();
            assert_eq!(
                dfa.run(bytes),
                (lo..=hi).contains(&cp),
                "range {lo:#x}..={hi:#x}, code point {cp:#x}"
            );
        }
    }
}

#[test]
fn a_dead_state_is_reported_at_the_first_dead_byte() {
    // "ab" then any byte, anything after that dead.
    let dfa = dfa_of(|nfa, s, a| {
        let t = nfa.bytes(s, b"ab").unwrap();
        nfa.range(t, 0, 255, a);
    });
    let mut w = DfaWalker::new(&dfa);
    assert_eq!(w.feed(&dfa, b"ab"), Verdict::Reject);
    assert_eq!(w.feed(&dfa, b"abc"), Verdict::Accept);
    assert_eq!(w.feed(&dfa, b"abcd"), Verdict::DeadAt(4));
    assert_eq!(w.feed(&dfa, b"ac"), Verdict::DeadAt(2));
    assert_eq!(w.feed(&dfa, b"acz"), Verdict::DeadAt(2));
    assert_eq!(w.feed(&dfa, b"abz"), Verdict::Accept);
    assert_eq!(w.feed(&dfa, b""), Verdict::Reject);
    assert_eq!(w.feed(&dfa, b"b"), Verdict::DeadAt(1));
}

#[test]
fn an_empty_language_starts_dead() {
    let dfa = dfa_of(|_, _, _| {});
    assert_eq!(dfa.start(), DEAD);
    assert!(!dfa.run(b""));
    let mut w = DfaWalker::new(&dfa);
    assert_eq!(w.feed(&dfa, b"x"), Verdict::DeadAt(0));
}

#[test]
fn complement_ranges_is_the_exact_complement() {
    assert_eq!(complement_ranges(&[]), vec![(0, MAX_CODE_POINT)]);
    assert_eq!(
        complement_ranges(&[(0x61, 0x7A), (0x30, 0x39)]),
        vec![(0, 0x2F), (0x3A, 0x60), (0x7B, MAX_CODE_POINT)]
    );
    assert_eq!(complement_ranges(&[(0, MAX_CODE_POINT)]), vec![]);
    assert_eq!(
        complement_ranges(&[(0, 5), (3, 9)]),
        vec![(10, MAX_CODE_POINT)]
    );
}

/// Random terms over an alphabet that exercises ASCII, multi-byte UTF-8
/// and ill-formed bytes, so every superset path is reached.
fn random_terms(seed: u64, n: usize) -> Vec<Vec<u8>> {
    let mut s = seed;
    let alphabet: &[&[u8]] = &[
        b"a",
        b"b",
        b"c",
        b"t",
        b"1",
        b"2",
        b"9",
        b"0",
        b"x",
        b"-",
        b".",
        b"_",
        "é".as_bytes(),
        "€".as_bytes(),
        "𝄞".as_bytes(),
        &[0xFF],
        &[0xC3],
        &[0x80],
    ];
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let len = (s % 7) as usize;
            let mut t = Vec::new();
            let mut x = s;
            for _ in 0..len {
                x = x.rotate_left(9) ^ 0x9E37_79B9;
                t.extend_from_slice(alphabet[(x % alphabet.len() as u64) as usize]);
            }
            t
        })
        .collect()
}

/// The contract every caller relies on: whatever the exact matcher
/// accepts, the DFA accepts -- and a prefix the DFA calls dead has no
/// accepted extension among the sample.
fn assert_superset(dfa: &ByteDfa, terms: &[Vec<u8>], exact: impl Fn(&[u8]) -> bool, what: &str) {
    // The precomputed live runs (`prune` builds them from the range
    // transitions) against a scan of the table itself, for every state and
    // byte: the next live byte, and runs that are maximal and ordered.
    for s in 0..dfa.num_states() as u32 {
        for after in 0..=255u8 {
            let want = (after.saturating_add(1)..=255)
                .filter(|_| after < 255)
                .find(|&b| dfa.step(s, b) != DEAD);
            assert_eq!(
                dfa.next_live_byte(s, after),
                want,
                "{what}: state {s} after {after}"
            );
        }
        let (lo, hi) = (
            dfa.live_offsets[s as usize] as usize,
            dfa.live_offsets[s as usize + 1] as usize,
        );
        for pair in dfa.live_ranges[lo..hi].windows(2) {
            assert!(
                pair[0].1 as usize + 1 < pair[1].0 as usize,
                "{what}: state {s} runs {pair:?} not maximal"
            );
        }
    }
    assert_eq!(dfa.live_offsets.len(), dfa.num_states() + 1, "{what}");

    let mut sorted = terms.to_vec();
    sorted.sort();
    sorted.dedup();
    let mut w = DfaWalker::new(dfa);
    for t in &sorted {
        let verdict = w.feed(dfa, t);
        if exact(t) {
            assert_eq!(verdict, Verdict::Accept, "{what}: exact accepts {t:?}");
        }
        if let Verdict::DeadAt(k) = verdict {
            for u in sorted.iter().filter(|u| u.starts_with(&t[..k])) {
                assert!(
                    !exact(u),
                    "{what}: prefix {:?} dead but {u:?} matches",
                    &t[..k]
                );
            }
        }
    }
}

#[test]
fn regexp_automata_accept_a_superset_and_are_exact_for_plain_patterns() {
    let terms = random_terms(0xABCDEF, 4000);
    for (pattern, exact_expected) in [
        ("t1[0-9]", true),
        ("t.9", true),
        ("[a-c]+", true),
        ("(ab|t)*1", true),
        ("[^a]x?", true),
        ("a{2,3}b", true),
        ("é.", true),
        ("#", true),
        ("@", true),
        ("\"ab\"c", true),
        ("[a-c]+&.*b.*", true),
        ("t<1-99>", true),
        ("..€", true),
    ] {
        let p = crate::regexp::RegexpPattern::parse(pattern).expect("parse");
        let dfa = p.to_dfa().expect("dfa");
        assert_superset(&dfa, &terms, |t| p.matches(t), pattern);
        if exact_expected {
            for t in &terms {
                assert_eq!(dfa.run(t), p.matches(t), "{pattern} exact on {t:?}");
            }
        }
    }
}

/// Intersection and numeric intervals compile to exact automata: every
/// digit string up to four wide, with and without leading zeros and with
/// a letter either side, gets the same answer from the DFA as from the
/// exact matcher.
#[test]
fn regexp_dfas_are_exact_for_intersection_and_intervals() {
    let mut terms: Vec<Vec<u8>> = Vec::new();
    for n in 0..10_000u32 {
        for w in 1..=5usize {
            let s = format!("{n:0w$}");
            if s.len() == w {
                terms.push(s.clone().into_bytes());
                terms.push(format!("t{s}").into_bytes());
                terms.push(format!("{s}x").into_bytes());
            }
        }
    }
    terms.extend(random_terms(0xFEED, 3000));
    for pattern in [
        "<0-0>",
        "<0-9>",
        "<5-123>",
        "<1-99>",
        "<007-120>",
        "<00-99>",
        "<10-9999>",
        "<0-2147483647>",
        "<2000000000-2147483647>",
        "<999-1000>",
        "t<1-99>",
        "<1-12>x",
        "<12-12>",
        "[a-c]+&.*b.*",
        "t.*&.*9",
        "(t1|t2).*&.*[0-4]",
        "<0-500>&<250-1000>",
        "#&.*",
        ".*&.*",
        "[0-9]{3}&<100-199>",
    ] {
        let p = crate::regexp::RegexpPattern::parse(pattern).expect("parse");
        let dfa = p.to_dfa().expect("dfa");
        for t in &terms {
            assert_eq!(
                dfa.run(t),
                p.matches(t),
                "{pattern} on {:?}",
                String::from_utf8_lossy(t)
            );
        }
    }
}

#[test]
fn wildcard_automata_accept_a_superset_and_are_exact_on_ascii() {
    let terms = random_terms(0x1234, 4000);
    for pattern in [
        &b"t?9"[..],
        b"t*",
        b"*1",
        b"a?c*",
        b"?",
        b"\\*x",
        b"*",
        b"t??",
    ] {
        let p = crate::wildcard::WildcardPattern::new(pattern);
        let dfa = p.to_dfa().expect("dfa");
        assert_superset(
            &dfa,
            &terms,
            |t| p.matches(t),
            &String::from_utf8_lossy(pattern),
        );
        for t in terms.iter().filter(|t| t.is_ascii()) {
            assert_eq!(dfa.run(t), p.matches(t), "{pattern:?} exact on ascii {t:?}");
        }
    }
}

#[test]
fn levenshtein_automata_accept_every_term_within_the_budget() {
    let terms = random_terms(0x77, 6000);
    for (target, edits, prefix, transpositions) in [
        (&b"t123"[..], 2u8, 0usize, true),
        (b"t12", 1, 0, true),
        (b"abc", 2, 1, false),
        ("té€".as_bytes(), 1, 0, true),
        (b"ba", 2, 0, true),
        (b"", 1, 0, true),
        // the prefix covering the whole term: an empty suffix
        (b"ab", 1, 2, true),
        (b"ab", 2, 5, false),
        // no edits: the exact term behind its prefix
        (b"abc", 0, 1, true),
        // non-ASCII at two edits: edges sharing a UTF-8 lead byte, the
        // subset-construction branch of `from_byte_automaton`
        ("é€x".as_bytes(), 2, 1, true),
    ] {
        let m = crate::fuzzy::FuzzyMatch::new(target, edits, prefix, transpositions);
        let dfa = m.to_dfa(edits).expect("dfa");
        // Lucene's automaton reads UTF-8: a term that is not valid UTF-8
        // never matches, whatever its lossy decoding is within reach of
        assert_superset(
            &dfa,
            &terms,
            |t| m.edits_within(t, edits).is_some(),
            &format!("{:?}~{edits}", String::from_utf8_lossy(target)),
        );
        for t in terms.iter().filter(|t| std::str::from_utf8(t).is_err()) {
            assert!(!dfa.run(t), "{target:?}~{edits} accepts ill-formed {t:?}");
        }
        // Lucene's parametric automaton is exact, not just a superset:
        // it accepts what the banded distance accepts, and nothing else
        for t in &terms {
            assert_eq!(
                dfa.run(t),
                m.edits_within(t, edits).is_some(),
                "{target:?}~{edits} prefix {prefix} on {t:?}"
            );
        }
    }
}

#[test]
fn construction_limits_refuse_rather_than_blow_up() {
    let mut nfa = Nfa::new();
    let mut last = None;
    for _ in 0..=MAX_NFA_STATES {
        last = nfa.state();
    }
    assert!(last.is_none());
}
