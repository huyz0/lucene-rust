#![cfg(test)]
use super::*;

/// **A stream of distinct patterns cannot grow the process-wide automaton
/// cache past its bound.** Every distinct user pattern lands in it, so a
/// workload of unique wildcard queries is exactly the tight loop that
/// would grow an unbounded one. Three times the bound, checked after each.
#[test]
fn the_process_wide_automaton_cache_stays_bounded_under_distinct_patterns() {
    for i in 0..3 * MAX_CACHED_PATTERNS {
        WildcardPattern::new(format!("bound{i}*x?").as_bytes()).to_dfa_cached();
        let len = DFA_CACHE.lock().unwrap_or_else(|e| e.into_inner()).len();
        assert!(len <= MAX_CACHED_PATTERNS, "pattern {i}: {len} cached");
    }
}

fn m(pattern: &str, term: &str) -> bool {
    WildcardPattern::new(pattern.as_bytes()).matches(term.as_bytes())
}

#[test]
fn empty_pattern_matches_only_empty_term() {
    assert!(m("", ""));
    assert!(!m("", "a"));
}

#[test]
fn pure_star_matches_everything_including_empty() {
    assert!(m("*", ""));
    assert!(m("*", "anything"));
    assert!(m("*", "with spaces and \0 bytes"));
}

#[test]
fn literal_pattern_matches_only_itself() {
    assert!(m("term", "term"));
    assert!(!m("term", "terms"));
    assert!(!m("term", "ter"));
    assert!(!m("term", "TERM"));
}

#[test]
fn question_mark_matches_exactly_one_ascii_byte() {
    assert!(m("a?c", "abc"));
    assert!(m("a?c", "axc"));
    assert!(!m("a?c", "ac"));
    assert!(!m("a?c", "abbc"));
    assert!(!m("?", ""));
    assert!(m("?", "x"));
}

#[test]
fn question_mark_matches_one_multi_byte_utf8_codepoint() {
    // Real Lucene's WildcardQuery.toAutomaton maps `?` to one Unicode
    // codepoint via codePointAt/charCount, not one encoded byte -- "a?"
    // must match "a€" (€ = 3 UTF-8 bytes: E2 82 AC) as a single `?`,
    // consuming the whole character, not just its first byte.
    assert!(m("a?", "a€"));
    assert!(!m("a?", "a€x")); // one extra byte left over after the `?`
    assert!(m("a?c", "a€c"));
    assert!(m("?", "𐍈")); // 4-byte codepoint (U+10348)
    assert!(m("?", "é")); // 2-byte codepoint
}

#[test]
fn question_mark_falls_back_to_one_byte_on_invalid_utf8() {
    // A `?` positioned at a non-UTF-8 byte (continuation byte or invalid
    // lead byte) has no codepoint to align to, so it falls back to
    // consuming exactly 1 byte -- confirms the fallback path in
    // `utf8_codepoint_len` rather than panicking or misbehaving.
    let p = WildcardPattern::new(b"a?c");
    assert!(p.matches(&[b'a', 0x80, b'c'])); // 0x80: bare continuation byte
    assert!(p.matches(&[b'a', 0xFF, b'c'])); // 0xFF: never a valid UTF-8 byte
}

#[test]
fn star_suffix_is_prefix_matching() {
    assert!(m("term*", "term"));
    assert!(m("term*", "term0000"));
    assert!(m("term*", "termxyz"));
    assert!(!m("term*", "ter"));
    assert!(!m("term*", "xterm"));
}

#[test]
fn star_prefix_is_suffix_matching() {
    assert!(m("*0000", "term0000"));
    assert!(m("*0000", "0000"));
    assert!(!m("*0000", "0000x"));
}

#[test]
fn star_in_the_middle() {
    assert!(m("term0*99", "term00099"));
    assert!(m("term0*99", "term099"));
    assert!(!m("term0*99", "term098"));
}

#[test]
fn adjacent_stars_collapse_like_one_star() {
    assert!(m("**", "anything"));
    assert!(m("a**b", "ab"));
    assert!(m("a**b", "axxxb"));
    assert!(!m("a**b", "ba"));
}

#[test]
fn matches_nothing_when_no_term_can_satisfy_it() {
    assert!(!m("a?c", "ac"));
    assert!(!m("literal", "totally different"));
    assert!(!m("prefix*suffix", "prefix-only"));
}

#[test]
fn prefix_constructor_matches_prefix_query_semantics() {
    let p = WildcardPattern::prefix(b"term");
    assert!(p.matches(b"term"));
    assert!(p.matches(b"term0000"));
    assert!(!p.matches(b"ter"));
    assert!(!p.matches(b"xterm"));
}

#[test]
fn literal_prefix_extraction() {
    assert_eq!(
        WildcardPattern::new(b"foo*ba?").literal_prefix(),
        b"foo".to_vec()
    );
    assert_eq!(WildcardPattern::new(b"*abc").literal_prefix(), b"".to_vec());
    assert_eq!(
        WildcardPattern::new(b"abc").literal_prefix(),
        b"abc".to_vec()
    );
    assert_eq!(WildcardPattern::new(b"?abc").literal_prefix(), b"".to_vec());
    assert_eq!(WildcardPattern::new(b"").literal_prefix(), b"".to_vec());
}

#[test]
fn escaped_star_and_question_mark_match_only_the_literal_character() {
    assert!(m(r"a\*b", "a*b"));
    assert!(!m(r"a\*b", "axb"));
    assert!(!m(r"a\*b", "ab"));
    assert!(m(r"a\?b", "a?b"));
    assert!(!m(r"a\?b", "axb"));
}

#[test]
fn escaped_backslash_matches_a_literal_backslash() {
    assert!(m(r"a\\b", r"a\b"));
    assert!(!m(r"a\\b", "ab"));
}

#[test]
fn trailing_unescaped_backslash_matches_itself_literally() {
    assert!(m(r"abc\", r"abc\"));
    assert!(!m(r"abc\", "abc"));
}

#[test]
fn escaping_a_non_special_byte_is_a_harmless_no_op() {
    assert!(m(r"a\bc", "abc"));
}

#[test]
fn matches_bytes_that_are_not_valid_utf8() {
    // Terms are arbitrary bytes, not guaranteed UTF-8 -- confirm the
    // matcher operates byte-wise and doesn't assume valid UTF-8.
    let p = WildcardPattern::new(&[b'a', b'*', 0xFF]);
    assert!(p.matches(&[b'a', 0x80, 0xFF]));
    assert!(!p.matches(&[b'a', 0x80, 0xFE]));
}
