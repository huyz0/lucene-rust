#![cfg(test)]
use super::*;

fn finds(p: &str, s: &str) -> Vec<(i32, i32)> {
    let p = JavaPattern::compile(p).unwrap();
    let mut m = JavaMatcher::new(&p, s);
    let mut out = Vec::new();
    while m.find() {
        out.push((m.start(0), m.end(0)));
    }
    out
}

fn rejected(p: &str) -> String {
    match JavaPattern::compile(p) {
        Err(AnalysisError::IllegalArgument(m)) => m,
        other => panic!("{p} compiled: {other:?}"),
    }
}

#[test]
fn java_classes_are_ascii() {
    let p = JavaPattern::compile("\\w+").unwrap();
    assert!(p.matches("abc_9"));
    assert!(!p.matches("é"));
    let p = JavaPattern::compile("[\\d\\s]+").unwrap();
    assert!(p.matches("1 2\t3\u{b}"));
    assert!(!p.matches("١"));
    let p = JavaPattern::compile("\\D\\W\\S").unwrap();
    assert!(p.matches("a-x"));
    let p = JavaPattern::compile("[\\W]").unwrap();
    assert!(p.matches("é") && !p.matches("a"));
    let p = JavaPattern::compile("[^\\W]").unwrap();
    assert!(p.matches("a") && !p.matches("é"));
    let p = JavaPattern::compile("[]a]+").unwrap();
    assert!(p.matches("]a]"));
    let p = JavaPattern::compile("[^]a]").unwrap();
    assert!(p.matches("b") && !p.matches("]"));
    assert!(JavaPattern::compile("\\.").unwrap().matches("."));
    assert!(JavaPattern::compile("(a").is_err());
    assert!(JavaPattern::compile("a\\").is_err());
    let never = JavaPattern::compile("[^\\x{0}-\\x{10FFFF}]").unwrap();
    assert!(!never.find_in("a\u{10FFFF}") && !never.matches(""));
    for (p, yes, no) in [
        ("\\p{Alpha}", "a", "é"),
        ("\\p{Upper}", "A", "É"),
        ("\\p{Lower}", "a", "é"),
        ("\\p{Punct}", "!", "¡"),
        ("\\p{Space}", "\u{b}", "\u{a0}"),
        ("\\p{Digit}", "1", "١"),
        ("\\p{Alnum}", "1", "é"),
        ("\\p{Graph}", "~", " "),
        ("\\p{Print}", " ", "\u{7f}"),
        ("\\p{Blank}", "\t", "\n"),
        ("\\p{Cntrl}", "\u{7f}", "a"),
        ("\\p{XDigit}", "f", "g"),
        ("\\p{ASCII}", "\u{7f}", "é"),
        ("\\p{L}", "é", "1"),
        ("\\pL", "é", "1"),
        ("\\p{IsL}", "é", "1"),
        ("\\p{IsLu}", "É", "é"),
        ("\\p{gc=Nd}", "١", "a"),
        ("\\p{general_category=Lu}", "A", "a"),
        ("\\p{LC}", "ǅ", "ʰ"),
        ("\\p{IsLC}", "ǅ", "ʰ"),
        ("\\p{LD}", "١", "_"),
        ("\\p{L1}", "ÿ", "Ā"),
        ("\\p{all}", "\u{10FFFF}", ""),
        ("\\P{L}", "1", "a"),
        ("[\\p{L}&&\\p{Lu}]", "A", "a"),
        ("[\\p{L}&&[^\\p{Lu}]]", "a", "A"),
        ("[a-z&&[^aeiou]]", "b", "a"),
        ("[^a-c&&b-d]", "a", "b"),
        ("[[:alpha:]]", ":", "b"),
        ("[[:^alpha:]]", "^", "b"),
        ("[x[:alpha:]]", "x", "b"),
        ("\\h", "\u{3000}", "\n"),
        ("\\H", "a", "\t"),
        ("\\v", "\u{2028}", " "),
        ("\\V", " ", "\u{b}"),
        ("[\\v]", "\u{85}", "a"),
        ("\\e", "\u{1b}", "e"),
        ("\\Qa.b\\E", "a.b", "axb"),
        ("[\\Q-]\\E]", "]", "a"),
        ("\\Qa", "a", "b"),
        ("\\uD83D\\uDE00", "😀", "a"),
        ("\\x{1F600}", "😀", "a"),
        ("\\u00e9", "é", "e"),
        ("\\<a\\>", "<a>", "a"),
        (".", "a", "\u{85}"),
        ("(?s).", "\u{2028}", ""),
        ("(?<ab1>x)", "x", "y"),
        ("a{2}", "aa", "a"),
        ("a{1,}", "aa", ""),
        ("a{0,1}?", "a", "aa"),
    ] {
        let c = JavaPattern::compile(p).unwrap();
        assert!(c.matches(yes), "{p} should match {yes:?}");
        assert!(!c.matches(no), "{p} should not match {no:?}");
    }
}

#[test]
fn case_insensitivity_is_javas() {
    for (p, s, want) in [
        ("(?i)a", "aA", 2),
        ("(?i)é", "éÉ", 1),
        ("(?iu)é", "éÉ", 2),
        ("(?i)[a-c]", "aBcD", 3),
        ("(?i)[é]", "éÉ", 1),
        ("(?iu)[é]", "éÉ", 2),
        ("(?i)k", "kK\u{212A}", 2),
        ("(?iu)k", "kK\u{212A}", 3),
        ("(?iu)\u{212A}", "kK", 2),
        ("(?i)[a-z]", "\u{212A}\u{17F}", 0),
        ("(?iu)[a-z]", "\u{212A}\u{17F}", 2),
        ("(?iu)[^k]", "\u{212A}", 0),
        ("(?i)[^a]", "aAb", 1),
        ("(?iu)[ÿ]", "Ÿ", 1),
        ("(?iu)[µ]", "μΜ", 2),
        ("(?iu)i", "\u{130}\u{131}I", 3),
        ("(?iu)[\u{130}]", "iI\u{131}", 3),
        ("(?iu)ß", "\u{1E9E}", 0),
        ("(?iu)ßx", "\u{1E9E}x", 1),
        ("(?iu)\u{1E9E}", "ß", 1),
        ("(?iu)[ß]", "\u{1E9E}", 0),
        ("(?iu)[à-å]", "Å\u{212B}", 2),
        ("(?i)[à-å]", "Å", 0),
        ("(?iu)[À]", "à", 1),
        ("(?i)[À]", "à", 0),
        ("(?iu)ǅ", "ǄǆǅX", 3),
        ("(?iu)[Ǆ-Ǆ]", "ǅǆ", 2),
        ("(?i)\\p{Lu}", "aAǅʰ", 3),
        ("(?i)\\P{Lu}", "aA1", 1),
        ("(?i)\\p{Upper}", "aAé", 2),
        ("(?iu)\\w", "\u{212A}", 0),
        ("(?i)[a&&b]", "A", 0),
        ("(?i)[a-c&&B]", "b", 1),
        ("(?i:a)b", "AbAB", 1),
        ("a(?i)b|c", "aBC", 2),
        ("(?i)(?-i:a)", "A", 0),
        ("(?i)(?u)k", "\u{212A}", 1),
        ("(?i)\\<A", "<a", 1),
    ] {
        assert_eq!(finds(p, s).len(), want, "{p} on {s:?}");
    }
}

#[test]
fn dollar_matches_before_a_final_line_terminator() {
    assert_eq!(finds("$", "a\n"), vec![(1, 1), (2, 2)]);
    assert_eq!(finds("$", "a\r\n"), vec![(1, 1), (3, 3)]);
    assert_eq!(finds("$", "a\r"), vec![(1, 1), (2, 2)]);
    assert_eq!(finds("$", "a\u{85}"), vec![(1, 1), (2, 2)]);
    assert_eq!(finds("$", "a\u{2028}"), vec![(1, 1), (2, 2)]);
    assert_eq!(finds("$", "a\n\n"), vec![(2, 2), (3, 3)]);
    assert_eq!(finds("$", "a"), vec![(1, 1)]);
    assert_eq!(finds("$", ""), vec![(0, 0)]);
    assert_eq!(finds("a$\\n", "a\n"), vec![(0, 2)]);
    assert_eq!(finds("\\s+", "a \n"), vec![(1, 3)]);
    assert_eq!(finds("\\n", "a\n"), vec![(1, 2)]);
    assert_eq!(finds("x*", "é\n"), vec![(0, 0), (1, 1), (2, 2)]);
    let p = JavaPattern::compile("^a$").unwrap();
    assert!(p.matches("a") && !p.matches("a\n") && p.find_in("a\n"));
    assert!(!p.find_in("a\nb"));
    let p = JavaPattern::compile("a$\\n").unwrap();
    assert!(p.matches("a\n"));
    assert_eq!(p.replace("a\r\n", "<$0>", true).unwrap(), "a\r\n");
    assert_eq!(
        JavaPattern::compile("(a)$")
            .unwrap()
            .replace("ba\n", "[$1]", true)
            .unwrap(),
        "b[a]\n"
    );
}

#[test]
fn rejects_what_differs_from_java() {
    // Java compiles these (`Pattern.compile`, JDK 21); the shim cannot.
    for p in [
        "\\bfox",
        "\\B",
        "(?m)^a",
        "(?m)a$",
        "(?x)a",
        "(?U)a",
        "[a~~b]",
        "[&&a]",
        "[a&&]",
        "[a&&&b]",
        "\\p{IsLatin}",
        "\\p{InGreek}",
        "\\p{IsAlphabetic}",
        "\\p{javaLowerCase}",
        "\\p{sc=Latin}",
        "(a|)*",
        "(a*)+",
        "(a?){2}",
        "a++",
        "a*+",
        "a**",
        "(?<=a)b",
        "(a)\\1",
        "\\Z",
        "\\G",
        "\\R",
        "\\X",
        "\\cA",
        "\\0101",
        "\\uD800",
        "\\N{LATIN SMALL LETTER A}",
        "(?>ab)",
    ] {
        let e = JavaPattern::compile_shim(p).unwrap_err();
        assert!(is_unsupported(&e), "{p}: {e}");
        // The backtracking matcher takes them over -- but for two it
        // refuses too, and one Java rejects.
        match (p, JavaPattern::compile(p)) {
            ("\\X" | "\\N{LATIN SMALL LETTER A}", Err(e)) => assert!(is_unsupported(&e), "{p}"),
            ("a**", Err(e)) => {
                assert!(e.to_string().contains("Dangling meta character"), "{e}")
            }
            (_, Ok(q)) => assert!(q.is_backtracking(), "{p}"),
            (_, Err(e)) => panic!("{p}: {e}"),
        }
    }
    // Java refuses these too; the `regex` crate alone would not.
    for p in [
        "\\b{start}",
        "(?R)a",
        "(?P<n>a)",
        "(?<a_b>a)",
        "\\u{e9}",
        "\\U000000E9",
        "a{ 2 }",
        "a{2, 3}",
        "[a--b]",
        "x{,3}",
    ] {
        let e = JavaPattern::compile(p).unwrap_err();
        assert!(!is_unsupported(&e), "{p}: {e}");
        assert!(rejected(p).contains("PatternSyntaxException"), "{p}");
    }
    // Java refuses these as well; the shim reports its own limit.
    for p in [
        "\\p{Latin}",
        "\\p{gc:Lu}",
        "\\p{Uppercase_Letter}",
        "\\p{ Lu }",
    ] {
        assert!(
            is_unsupported(&JavaPattern::compile_shim(p).unwrap_err()),
            "{p}"
        );
        assert!(
            rejected(p).contains("Unknown character property name"),
            "{p}"
        );
    }
    // A non-repeating or non-capturing nullable body is fine.
    for p in ["(a|)?", "(?:a|)*", "(a*)", "((a)|b)*"] {
        assert!(JavaPattern::compile(p).is_ok(), "{p}");
    }
}

/// Java's `find()`: after an empty match it moves one UTF-16 unit on --
/// which lands between the surrogates of a supplementary character.
#[test]
fn matcher_finds_like_java() {
    assert_eq!(
        finds("a*", "baa😀"),
        vec![(0, 0), (1, 3), (3, 3), (4, 4), (5, 5)]
    );
    assert_eq!(
        finds("x*", "abxd"),
        vec![(0, 0), (1, 1), (2, 3), (3, 3), (4, 4)]
    );
    let p = JavaPattern::compile("x*").unwrap();
    assert_eq!(p.replace("abxd", "-", true).unwrap(), "-a-b--d-");
    assert_eq!(p.replace("😀", "-", true).unwrap(), "-\u{FFFD}-\u{FFFD}-");
    assert_eq!(p.replace("abxd", "-", false).unwrap(), "-abxd");
    // A class holding the low surrogates matches the lone half there.
    assert_eq!(finds("\\A|.", "😀"), vec![(0, 0), (1, 2)]);
    let p = JavaPattern::compile("\\A|(.)").unwrap();
    let mut m = JavaMatcher::new(&p, "😀b");
    assert!(m.find());
    assert_eq!((m.start(0), m.end(0), m.start(1)), (0, 0, -1));
    assert!(m.find());
    assert_eq!((m.start(1), m.end(1)), (1, 2));
    assert_eq!(m.group(1).unwrap(), "\u{FFFD}");
    let mut units = Vec::new();
    m.slice_utf16_into(0, 2, &mut units);
    assert_eq!(units, vec![0xD83D, 0xDE00]);
    units.clear();
    m.slice_utf16_into(1, 3, &mut units);
    assert_eq!(units, vec![0xDE00, u16::from(b'b')]);
    units.clear();
    m.slice_utf16_into(0, 1, &mut units);
    assert_eq!(units, vec![0xD83D]);
    assert_eq!(m.slice(0, 1), "\u{FFFD}");
    assert_eq!(m.slice(0, 3), "😀b");
    assert_eq!(m.slice(2, 1), "");
    assert!(m.find());
    assert_eq!(m.group(0).unwrap(), "b");
    assert!(!m.find());
    // Groups that take part in the empty match between the surrogates.
    let p = JavaPattern::compile("(a*)(b)?").unwrap();
    let mut m = JavaMatcher::new(&p, "😀");
    assert!(m.find() && m.find());
    assert_eq!((m.start(0), m.start(1), m.start(2)), (1, 1, -1));
    assert_eq!(m.group(2), None);
    // A pattern that cannot match empty there skips the half.
    assert_eq!(finds("x*$", "😀"), vec![(2, 2)]);
    let p = JavaPattern::compile("(x)|(y)").unwrap();
    let mut m = JavaMatcher::new(&p, "😀y");
    assert_eq!(m.group_count(), 2);
    assert!(m.find());
    assert_eq!((m.start(1), m.end(1), m.start(2), m.end(2)), (-1, -1, 2, 3));
    assert_eq!(m.text(), "😀y");
    assert_eq!(m.len_utf16(), 3);
    assert!(!m.find());
    assert_eq!(m.start(0), -1, "a failed find clears the groups");
    assert!(!m.find(), "an exhausted matcher stays exhausted");
    // The sentinel and the mid-pair replay together.
    assert_eq!(finds("[^a]*$", "😀\n"), vec![(0, 3), (3, 3)]);
    assert_eq!(finds("\\A|[^a]$", "😀\n"), vec![(0, 0), (1, 2), (2, 3)]);
}

#[test]
fn replacement_syntax_is_javas() {
    let p = JavaPattern::compile("([a-z]+)-([a-z]+)").unwrap();
    assert_eq!(
        p.replace("ab-cd ef-gh", "$2_$1", true).unwrap(),
        "cd_ab gh_ef"
    );
    assert_eq!(
        p.replace("ab-cd ef-gh", "$2_$1", false).unwrap(),
        "cd_ab ef-gh"
    );
    assert_eq!(p.replace("ab-cd", "\\$1$10", true).unwrap(), "$1ab0");
    let n = JavaPattern::compile("(?<w>x)").unwrap();
    assert_eq!(n.replace("axb", "[${w}]", true).unwrap(), "a[x]b");
    assert!(n.replace("axb", "${v}", true).is_err());
    assert!(p.replace("ab-cd", "$9", true).is_err());
    assert!(p.replace("ab-cd", "$", true).is_err());
    assert!(p.replace("ab-cd", "$x", true).is_err());
    assert!(p.replace("ab-cd", "x\\", true).is_err());
    assert!(n.replace("x", "${w", true).is_err());
    assert_eq!(p.replace("none", "$1", true).unwrap(), "none");
    let o = JavaPattern::compile("(a)|b").unwrap();
    assert_eq!(o.replace("ab", "[$1]", true).unwrap(), "[a][]");
    let mut units: Vec<u16> = Vec::new();
    let mut m = JavaMatcher::new(&o, "a");
    assert!(m.find());
    m.expand_replacement("<$1é>", &mut units).unwrap();
    assert_eq!(String::from_utf16(&units).unwrap(), "<aé>");
}

/// A group under a zero-count quantifier still counts (the `regex`
/// crate drops it): `groupCount()` and `$1` are Java's (JDK 21 and 25).
#[test]
fn groups_under_a_zero_count_quantifier_count() {
    for (p, text, groups, rep) in [
        ("(a){0}b", "ab", 1, "a<>"),
        ("(a*){0}", "ab", 1, "<>a<>b<>"),
        ("((a)){0}", "ab", 2, "<>a<>b<>"),
        ("(?<n>a){0}b", "ab", 1, "a<>"),
        ("(a|ab){0}c", "abc", 1, "ab<>"),
        ("(?:(a)){0}b", "ab", 1, "a<>"),
        ("(a){0,0}b", "ab", 1, "a<>"),
        ("(a){0}?b", "ab", 1, "a<>"),
    ] {
        let pat = JavaPattern::compile(p).unwrap();
        assert_eq!(JavaMatcher::new(&pat, text).group_count(), groups, "{p}");
        assert_eq!(pat.replace(text, "<$1>", true).unwrap(), rep, "{p}");
    }
    // Without a capture, the shim keeps it.
    assert!(!JavaPattern::compile("(?:a){0}b").unwrap().is_backtracking());
}

/// A deep match grows the backtracking stacks; the matcher keeps no
/// more than [`Scratch::KEEP`] bytes of them for the next `find()`.
#[test]
fn matcher_keeps_small_stacks() {
    let p = JavaPattern::compile("(?=a)(a|b)*c").unwrap();
    assert!(p.is_backtracking());
    let mut m = JavaMatcher::new(&p, &"ab".repeat(50_000));
    assert!(!m.try_find().unwrap());
    assert!(
        m.bt_scratch.footprint() <= Scratch::KEEP,
        "{}",
        m.bt_scratch.footprint()
    );
    m.reset("abc");
    assert!(m.try_find().unwrap());
    assert_eq!((m.start(1), m.end(1)), (1, 2));
}

#[test]
fn case_index_covers_every_simple_mapping() {
    for c in CASED_LIMIT..=MAX_CP {
        assert_eq!((to_upper_case(c), to_lower_case(c)), (c, c), "{c:#x}");
    }
    assert!(CASE_INDEX.cased.len() > 2000);
}

#[test]
fn code_point_sets() {
    let a = CpSet::from_ranges(vec![(5, 9), (1, 2), (3, 3), (20, 30)]);
    assert_eq!(a.0, vec![(1, 3), (5, 9), (20, 30)]);
    assert_eq!(
        a.intersect(&CpSet::range(2, 25)).0,
        vec![(2, 3), (5, 9), (20, 25)]
    );
    assert_eq!(a.complement().0[0], (0, 0));
    assert_eq!(CpSet::all().complement(), CpSet::default());
    assert_eq!(CpSet::default().complement(), CpSet::all());
    assert!(a.contains(7) && !a.contains(4) && !a.contains(31));
    assert_eq!(prepass("abc"), "abc");
    assert_eq!(prepass("\\uD83D"), "\\uD83D");
    assert_eq!(prepass("a\\"), "a\\");
}

/// Replacement text past ASCII: literal runs, an escaped character and
/// a group reference between them.
#[test]
fn replacement_with_non_ascii() {
    let p = JavaPattern::compile("(b)").unwrap();
    assert_eq!(p.replace("abc", "é\\ü$1ñ", true).unwrap(), "aéübñc");
    assert!(p.replace("abc", "$é", true).is_err());
    let mut m = JavaMatcher::new(&p, "abc");
    assert!(m.find());
    let mut units = Vec::new();
    m.expand_replacement("é\\ü$1ñ", &mut units).unwrap();
    assert_eq!(String::from_utf16(&units).unwrap(), "éübñ");
}

/// `find()` from the start: Java's `StackOverflowError` is `Err` from
/// the `try_` form, `false` from the plain one.
#[test]
fn find_in_reports_overflow() {
    let p = JavaPattern::compile("(a|b)*\\1c").unwrap();
    let long = "a".repeat(300_000);
    assert!(p.try_find_in(&long).is_err());
    assert!(!p.find_in(&long));
    assert!(p.try_find_in("abbc").unwrap());
    let s = JavaPattern::compile("b$").unwrap();
    assert!(s.try_find_in("ab\n").unwrap());
    assert!(!s.try_find_in("ba").unwrap());
}
