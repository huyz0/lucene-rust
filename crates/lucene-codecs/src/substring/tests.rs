#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;

fn build(values: &[Option<&str>], shift: u8) -> (Vec<u8>, SubstringEntry) {
    let docs: Vec<i32> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_some())
        .map(|(d, _)| d as i32)
        .collect();
    let present: Vec<&str> = values.iter().flatten().copied().collect();
    let mut data = vec![0u8; 3];
    let meta = write_data(
        &mut data,
        &docs,
        |i| present[i].as_bytes(),
        values.len() as i32,
        shift,
    );
    let back = read_meta(&mut SliceInput::new(&meta)).unwrap();
    (data, back)
}

fn search(values: &[Option<&str>], shift: u8, m: &SubstringMatcher) -> Vec<i32> {
    let (data, entry) = build(values, shift);
    let index = SubstringIndex::new(&data, &entry).unwrap();
    let blocks: Vec<u32> = match index.candidate_blocks(m.grams()).unwrap() {
        Candidates::All => (0..index.num_blocks() as u32).collect(),
        Candidates::Blocks(b) => b,
    };
    let mut out = Vec::new();
    let mut buf = Block::new();
    let mut scratch = Vec::new();
    for b in blocks {
        index.read_block(i64::from(b), &mut buf).unwrap();
        m.collect(&buf, &mut out, &mut scratch);
    }
    out
}

fn brute(values: &[Option<&str>], m: &SubstringMatcher) -> Vec<i32> {
    let mut scratch = Vec::new();
    values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_some_and(|v| m.matches(v.as_bytes(), &mut scratch)))
        .map(|(d, _)| d as i32)
        .collect()
}

const VALUES: &[Option<&str>] = &[
    Some("Connection timeout on 10.0.0.1"),
    None,
    Some("java.net.SocketTimeoutException: read timed out"),
    Some("ok"),
    Some(""),
    Some("ERROR disk full"),
    Some("error: retry"),
    None,
    Some("abcabcabc"),
    Some("İstanbul KELVIN \u{212A}"),
    Some("tail ab"),
    Some("ab"),
    Some("x"),
];

#[test]
fn every_pattern_finds_what_a_scan_finds() {
    let patterns: Vec<SubstringMatcher> = vec![
        SubstringMatcher::contains("timeout", false),
        SubstringMatcher::contains("timeout", true),
        SubstringMatcher::contains("Timeout", false),
        SubstringMatcher::contains("error", false),
        SubstringMatcher::contains("error", true),
        SubstringMatcher::contains("ab", false),
        SubstringMatcher::contains("", false),
        SubstringMatcher::contains("istanbul", true),
        SubstringMatcher::contains("kelvin k", true),
        SubstringMatcher::contains("cabca", false),
        SubstringMatcher::exact("ok", false),
        SubstringMatcher::exact("OK", true),
        SubstringMatcher::exact("", false),
        SubstringMatcher::wildcard("*", false),
        SubstringMatcher::wildcard("ab*", false),
        SubstringMatcher::wildcard("*ab", false),
        SubstringMatcher::wildcard("?b", false),
        SubstringMatcher::wildcard("e*r*", true),
        SubstringMatcher::wildcard("*time?out*", false),
        SubstringMatcher::wildcard("*10.0.*.1", false),
        SubstringMatcher::regexp(".*[Tt]imeout.*", false).unwrap(),
        SubstringMatcher::regexp("error.*", true).unwrap(),
        SubstringMatcher::regexp("(ok|x)", false).unwrap(),
        SubstringMatcher::regexp(".*abc(abc)+", false).unwrap(),
        SubstringMatcher::regexp(".*TIME(d|out).*", true).unwrap(),
        SubstringMatcher::regexp(".*ab(c|x)a.*", false).unwrap(),
        SubstringMatcher::wildcard("*a?c*a*", false),
        SubstringMatcher::wildcard("*KELVIN*", true),
        SubstringMatcher::wildcard("*?l?i*", false),
    ];
    for shift in [0u8, 1, 2, 5] {
        for m in &patterns {
            assert_eq!(
                search(VALUES, shift, m),
                brute(VALUES, m),
                "{m:?} shift {shift}"
            );
        }
    }
}

/// The leftmost-first glob and the backtracking matcher agree on every
/// pattern over a small alphabet, ASCII and not.
#[test]
fn glob_agrees_with_backtracking() {
    let alphabet = ["a", "b", "ab", "é"];
    let mut x = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n as u64) as usize
    };
    for _ in 0..3000 {
        let mut p = String::new();
        for _ in 0..1 + next(6) {
            p.push_str(["*", "?", "a", "b", "ab", "é"][next(6)]);
        }
        let mut v = String::new();
        for _ in 0..next(9) {
            v.push_str(alphabet[next(alphabet.len())]);
        }
        let m = SubstringMatcher::wildcard(&p, false);
        let slow = WildcardPattern::new(p.as_bytes()).matches(v.as_bytes());
        assert_eq!(
            m.matches(v.as_bytes(), &mut Vec::new()),
            slow,
            "{p:?} {v:?}"
        );
    }
}

/// The regexp's prefilter run and early-stopping automaton change no
/// answer of `RegexpPattern::matches`.
#[test]
fn regexp_prefilter_and_automaton_agree_with_the_pattern() {
    for (p, ci) in [
        (".*[Tt]imeout.*", false),
        (".*FAILED password.*", true),
        ("(a|b)*abc.*", false),
        (".*blk_-?[0-9]+ term", false),
    ] {
        let m = SubstringMatcher::regexp(p, ci).unwrap();
        let parsed = RegexpPattern::parse(p).unwrap();
        let r = if ci {
            parsed.ascii_case_insensitive()
        } else {
            parsed
        };
        for &v in VALUES.iter().flatten() {
            let v = v.as_bytes();
            assert_eq!(m.matches(v, &mut Vec::new()), r.matches(v), "{p} {v:?}");
        }
    }
    let lit = |p: &str| RegexpPattern::parse(p).unwrap().required_literal();
    assert_eq!(lit(".*blk_-?[0-9]+ terminating"), b" terminating");
    assert_eq!(lit("(ab)+cde.*"), b"cde");
    assert_eq!(lit(".*(x|y)"), b"");
    assert_eq!(lit("ab&abc"), b"abc");
}

/// A case-insensitive regexp folds like Lucene 10.5's `CaseFolding`:
/// class characters, non-ASCII letters and the Kelvin and long-s signs,
/// through the index as through a scan.
#[test]
fn case_insensitive_regexp_folds_beyond_ascii() {
    let values: &[Option<&str>] = &[
        Some("TIMEOUT now"),
        Some("\u{212A}elvin scale"),
        Some("caf\u{C9} CAF\u{C9} open"),
        Some("STRA\u{1E9E}E 5"),
        Some("pa\u{17F}\u{17F}word failed"),
        Some("nothing here"),
    ];
    let cases = [
        (".*[t]imeout.*", vec![0]),
        ("kelvin.*", vec![1]),
        (".*caf\u{E9}.*", vec![2]),
        ("stra\u{DF}e.*", vec![3]),
        (".*password.*", vec![4]),
    ];
    for (p, want) in cases {
        let m = SubstringMatcher::regexp(p, true).unwrap();
        for shift in [0u8, 2] {
            assert_eq!(search(values, shift, &m), want, "{p} shift {shift}");
        }
        assert_eq!(brute(values, &m), want, "{p}");
    }
}

#[test]
fn case_insensitive_wildcard_lowers_value_and_pattern() {
    let m = SubstringMatcher::contains("TIMEOUT", true);
    let mut s = Vec::new();
    assert!(m.matches(b"read Timeout", &mut s));
    assert!(!SubstringMatcher::contains("TIMEOUT", false).matches(b"read Timeout", &mut s));
    // The Kelvin sign lower-cases to `k`.
    assert!(SubstringMatcher::contains("k", true).matches("\u{212A}".as_bytes(), &mut s));
}

#[test]
fn grams_fold_and_anchor() {
    let m = SubstringMatcher::wildcard("Ab*", false);
    assert_eq!(
        m.grams(),
        &[gram_key(&[BOUNDARY, u32::from(b'a'), u32::from(b'b')])]
    );
    assert!(SubstringMatcher::wildcard("*ab*", false).grams().is_empty());
    // Case-insensitive grams leave out non-ASCII bytes.
    assert!(SubstringMatcher::contains("éé", true).grams().is_empty());
    assert_eq!(SubstringMatcher::contains("éé", false).grams().len(), 2);
}

#[test]
fn postings_round_trip_as_list_and_bitset() {
    // 300 documents, one per block: "common" in five of every seven
    // blocks (a bitset), "rare" in a few (a list).
    let values: Vec<String> = (0..300)
        .map(|i| match i {
            _ if i % 97 == 0 => format!("rare common {i}"),
            _ if i % 7 == 4 || i % 7 == 5 => format!("other {i}"),
            _ => format!("common {i}"),
        })
        .collect();
    let opts: Vec<Option<&str>> = values.iter().map(|s| Some(s.as_str())).collect();
    let (data, entry) = build(&opts, 0);
    let index = SubstringIndex::new(&data, &entry).unwrap();
    let common = gram_key(&[u32::from(b'c'), u32::from(b'o'), u32::from(b'm')]);
    let rare = gram_key(&[u32::from(b'r'), u32::from(b'a'), u32::from(b'r')]);
    let (ci, cdf) = index.find_gram(common).unwrap();
    let (ri, rdf) = index.find_gram(rare).unwrap();
    let commons: Vec<u32> = (0..300)
        .filter(|i| i % 97 == 0 || !(i % 7 == 4 || i % 7 == 5))
        .collect();
    assert_eq!((cdf as usize, rdf), (commons.len(), 4));
    assert!(index.bitset(ci).unwrap().is_some());
    assert!(index.bitset(ri).unwrap().is_none());
    let mut out = Vec::new();
    index.decode(ri, &mut out).unwrap();
    assert_eq!(out, vec![0, 97, 194, 291]);
    index.decode(ci, &mut out).unwrap();
    assert_eq!(out, commons);
    assert_eq!(
        index.candidate_blocks(&[common, rare]).unwrap(),
        Candidates::Blocks(vec![0, 97, 194, 291])
    );
    assert_eq!(
        index
            .candidate_blocks(&[common, gram_key(&[1, 2, 3])])
            .unwrap(),
        Candidates::Blocks(vec![])
    );
    let mut buf = Block::new();
    assert_eq!(
        index.value(97, &mut buf).unwrap(),
        Some(&b"rare common 97"[..])
    );
    assert_eq!(index.value(6, &mut buf).unwrap(), Some(&b"common 6"[..]));
}

#[test]
fn empty_and_sparse_fields() {
    let (data, entry) = build(&[None, None, Some("a"), None], 1);
    let index = SubstringIndex::new(&data, &entry).unwrap();
    let mut buf = Block::new();
    assert_eq!(index.value(0, &mut buf).unwrap(), None);
    assert_eq!(index.value(2, &mut buf).unwrap(), Some(&b"a"[..]));
    assert_eq!(index.value(3, &mut buf).unwrap(), None);
    assert!(index.read_block(9, &mut buf).is_err());
    let (data, entry) = build(&[], 5);
    let index = SubstringIndex::new(&data, &entry).unwrap();
    assert_eq!(index.num_blocks(), 0);
}

#[test]
fn corrupt_meta_is_refused() {
    let mut meta = vec![MAX_BLOCK_SHIFT + 1];
    meta.extend_from_slice(&[0; 64]);
    assert!(read_meta(&mut SliceInput::new(&meta)).is_err());
}
