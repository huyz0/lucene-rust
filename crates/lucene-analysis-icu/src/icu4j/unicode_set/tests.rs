#![cfg(test)]
use super::*;

/// `add_range` against a plain membership model, over many small and
/// overlapping, adjacent and disjoint additions.
#[test]
fn add_range_matches_a_model() {
    let mut set = UnicodeSet::new();
    let mut model = vec![false; 400];
    let mut x: u32 = 12345;
    for _ in 0..2000 {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let a = (x >> 8) % 390;
        let len = (x >> 20) % 7;
        let b = a + len;
        set.add_range(a, b);
        for m in &mut model[a as usize..=b as usize] {
            *m = true;
        }
        let ranges = set.ranges();
        assert!(ranges.windows(2).all(|w| w[0].1 + 1 < w[1].0), "{ranges:?}");
        for (c, &m) in model.iter().enumerate() {
            assert_eq!(set.contains(c as i32), m, "{c}");
        }
        if set.ranges().len() == 1 {
            break;
        }
    }
    set.add_range(5, 3);
    set.add_range(MAX_VALUE + 1, MAX_VALUE + 2);
    set.add_range(MAX_VALUE - 1, u32::MAX);
    assert_eq!(set.ranges().last(), Some(&(MAX_VALUE - 1, MAX_VALUE)));
}

fn set(p: &str) -> UnicodeSet {
    UnicodeSet::from_pattern(p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn err(p: &str) -> IcuError {
    UnicodeSet::from_pattern(p).expect_err(p)
}

/// Variables: `$vowel` = `[aeiou]` (as a stand-in U+F000), `$xy` = "xy",
/// `$other` names a non-set matcher (U+F001), `$empty` is empty.
struct Symbols {
    vowels: UnicodeSet,
}

impl SymbolTable for Symbols {
    fn lookup(&self, name: &str) -> Option<Vec<u16>> {
        match name {
            "vowel" => Some(vec![0xf000]),
            "xy" => Some(utf16::units("xy")),
            "other" => Some(vec![0xf001]),
            "empty" => Some(Vec::new()),
            _ => None,
        }
    }

    fn lookup_matcher(&self, c: i32) -> Option<SymbolMatcher<'_>> {
        match c {
            0xf000 => Some(SymbolMatcher::Set(&self.vowels)),
            0xf001 => Some(SymbolMatcher::Other),
            _ => None,
        }
    }

    fn parse_reference(&self, text: &[u16], pos: usize, limit: usize) -> Option<(String, usize)> {
        let mut i = pos;
        while i < limit && (text[i] as u8 as char).is_ascii_alphabetic() && text[i] < 0x80 {
            i = i.saturating_add(1);
        }
        (i > pos).then(|| (utf16::string(&text[pos..i]), i))
    }
}

#[test]
fn symbol_tables() {
    let sym = Symbols {
        vowels: set("[aeiou]"),
    };
    let parse = |p: &str| -> Result<(UnicodeSet, usize), IcuError> {
        let t = utf16::units(p);
        let mut pos = 0;
        let s = UnicodeSet::parse_at(&t, &mut pos, Some(&sym))?;
        Ok((s, pos))
    };
    let (s, end) = parse("[$vowel z] rest").unwrap();
    assert_eq!(s, set("[aeiouz]"));
    assert_eq!(end, 11);
    assert_eq!(parse("[$xy]").unwrap().0, set("[xy]"));
    assert_eq!(parse("[a$]").unwrap().0, set("[a\\uffff]"));
    assert!(parse("[$nope]").is_err());
    assert!(parse("[$other]").is_err());
    assert!(parse("[$empty]").is_err());
    assert!(parse("[a-$]").is_err());
    // A variable value that does not close its set inside it.
    assert!(parse("[[:L:]$xy").is_err());
    assert_eq!(parse("[[:Lu:]&[A-C]]").unwrap().0, set("[A-C]"));
    assert_eq!(parse("\\p{Lu}").unwrap().0, set("[:Lu:]"));
    assert!(UnicodeSet::resembles_pattern(&utf16::units("[a]"), 0));
    assert!(UnicodeSet::resembles_pattern(&utf16::units("\\p{L}"), 0));
    assert!(!UnicodeSet::resembles_pattern(&utf16::units("ab"), 0));
    assert!(!UnicodeSet::resembles_pattern(&utf16::units("\\p"), 0));
}

#[test]
fn set_queries() {
    let a = set("[a-c{xy}]");
    assert_eq!(a.size(), 4);
    assert!(a.contains_some(&set("[c-e]")));
    assert!(a.contains_none(&set("[d-e]")));
    assert!(a.contains_some(&set("[{xy}]")));
    assert!(a.matches_index_value(0x61));
    assert!(!a.matches_index_value(0x64));
    assert!(set("[\\u00fe-\\u0101]").matches_index_value(0x01));
    assert!(set("[{\\u0178z}]").matches_index_value(0x78));
    let mut b = UnicodeSet::new();
    b.add_all_code_points(&utf16::units("b\u{1F600}"));
    assert_eq!(b, set("[b\\U0001F600]"));
    let p = set("[a-c\\u0300{xy}\\U0001F600]");
    assert_eq!(set(&p.to_pattern()), p);
    let mut c = set("[a c-e x]");
    c.add_all(&set("[b f-g z]"));
    assert_eq!(c, set("[a-gxz]"));
}

#[test]
fn literals_ranges_and_negation() {
    let s = set("[a-c x]");
    assert_eq!(s.ranges(), &[(0x61, 0x63), (0x78, 0x78)]);
    let s = set("[^a-c]");
    assert!(!s.contains('b' as i32) && s.contains('d' as i32) && s.contains(0x10ffff));
    assert!(set("[-a]").contains('-' as i32));
    assert!(set("[^-a]").contains('b' as i32));
    assert!(set("[a-z-]").contains('-' as i32));
    assert!(set("[\\u0041\\x42\\x{43}\\U00000044\\105]").ranges() == [(0x41, 0x45)]);
    assert!(set("[\\t\\n]").contains(9));
    assert!(set("[\\cA]").contains(1));
    assert!(set("[\\-]").contains('-' as i32));
    assert!(set("[\\uD801\\uDC00]").contains(0x10400));
    assert!(set("[a$]").contains(0xffff));
    assert!(set("[$a]").contains('$' as i32));
    assert!(set("[$x]").contains('x' as i32));
    assert!(set("[{abc}]").has_strings());
    assert!(set("[{a}]").contains('a' as i32));
    assert!(set("[{a}-{c}]").contains('b' as i32));
    assert!(set("[a-b{xy}c]").has_strings());
    assert!(set("  [ a ]  ").contains('a' as i32));
    assert!(set("[]").is_empty());
}

#[test]
fn operators_and_nesting() {
    let s = set("[[a-z]-[aeiou]]");
    assert!(s.contains('b' as i32) && !s.contains('e' as i32));
    let s = set("[[a-z]&[c-e]]");
    assert_eq!(s.ranges(), &[(0x63, 0x65)]);
    let s = set("[[a-c][x-z]]");
    assert_eq!(s.ranges(), &[(0x61, 0x63), (0x78, 0x7a)]);
    let s = set("[a[x]]");
    assert!(s.contains('a' as i32));
    assert!(set("[[a][b]-]").contains('-' as i32));
}

#[test]
fn properties() {
    assert!(set("[:Latin:]").contains('a' as i32));
    assert!(set("[:^Latin:]").contains('1' as i32));
    assert!(set("\\p{Lu}").contains('A' as i32));
    assert!(!set("\\P{Lu}").contains('A' as i32));
    assert!(set("\\p{gc=Lu}").contains('A' as i32));
    assert!(set("\\p{General_Category=Letter}").contains('a' as i32));
    assert!(set("[\\p{sc=Grek}]").contains(0x3b1));
    assert!(set("[\\p{scx=Hani}]").contains(0x3001));
    assert!(set("[:Nonspacing Mark:]").contains(0x301));
    assert!(set("[:White_Space:]").contains(' ' as i32));
    assert!(!set("[:White_Space=No:]").contains(' ' as i32));
    assert!(set("[:lb=SA:]").contains(0xe01));
    assert!(set("[:ccc=230:]").contains(0x301));
    assert!(set("[:ANY:]").contains(0x10ffff));
    assert!(set("[:ascii:]").contains(0x7f));
    assert!(!set("[:Assigned:]").contains(0x10ffff));
    assert!(set("[[:Emoji:][:Extended_Pictographic:]]").contains(0x1f600));
    assert!(set("[[:Thai:]&[:LineBreak=SA:]]").contains(0xe01));
    assert!(set("[:ccc=Above:]").contains(0x301));
}

#[test]
fn errors() {
    for p in [
        "a",
        "[a",
        "[a-]x",
        "[z-a]",
        "[a-[b]]",
        "[[a]&]",
        "[a&[b]]",
        "[^^]",
        "[{ab]",
        "[[a]-b]",
        "[{ab}-c]",
        "[a-{bc}]",
        "[[a]&{b}]",
        "[[:Latin:]&]",
        "\\p{Lu",
        "\\p{NoSuchThing}",
        "\\p{gc=Nope}",
        "\\p{Nope=x}",
        "[:Line_Break:]",
        "[:ccc=300:]",
        "[:ccc=x1:]",
        "[\\",
        "[\\u12]",
        "[\\x{12]",
        "[\\U00110000]",
        "[a] b",
        "[:L:",
        "\\px",
        "[[a]-$]",
    ] {
        let e = err(p);
        assert!(
            matches!(
                e.kind(),
                crate::IcuErrorKind::IllegalArgument | crate::IcuErrorKind::IllegalIcuArgument
            ),
            "{p}: {e}"
        );
    }
    // a character name is a one-character set; a name no character
    // has is refused
    assert_eq!(set("[\\N{LATIN SMALL LETTER A}]").ranges(), &[(97, 97)]);
    assert_eq!(set("\\N{ latin  small letter a }").ranges(), &[(97, 97)]);
    assert_eq!(set("[:name=HYPHEN-MINUS:]").ranges(), &[(45, 45)]);
    assert_eq!(set("\\P{name=HYPHEN-MINUS}").size(), 0x10ffff);
    for p in ["\\N{NO SUCH CHARACTER}", "\\p{name=x}"] {
        let e = err(p);
        assert_eq!(e.message(), "Invalid character name", "{p}");
    }
    for p in ["[{ab}-{ad}]", "\\p{ID_Type=x}"] {
        assert_eq!(
            err(p).kind(),
            crate::IcuErrorKind::UnsupportedOperation,
            "{p}"
        );
    }
    let mut deep = "[".repeat(102);
    deep.push_str(&"]".repeat(102));
    assert!(err(&deep).message().contains("nested"));
}

#[test]
fn set_algebra_and_spans() {
    let mut s = UnicodeSet::from_range(0x61, 0x7a);
    s.add_range(0x30, 0x39);
    s.add_range(0x3a, 0x40);
    s.add_range(5, 2);
    s.add_range(0x200000, 0x200001);
    assert_eq!(s.ranges(), &[(0x30, 0x40), (0x61, 0x7a)]);
    let mut t = s.clone();
    t.remove_all(&UnicodeSet::from_range(0x35, 0x62));
    assert_eq!(t.ranges(), &[(0x30, 0x34), (0x63, 0x7a)]);
    t.add_string(&utf16::units("xy"));
    let mut u = t.clone();
    u.retain_all(&s);
    assert!(!u.has_strings());
    u.add_all(&t);
    assert!(u.has_strings());
    u.remove_all(&t);
    assert!(!u.has_strings());
    u.add_string(&utf16::units("\u{10400}"));
    assert!(u.contains(0x10400) && !u.contains(-1));
    let text = utf16::units("abc123\u{10400}x");
    let s = UnicodeSet::from_range(0x61, 0x7a);
    assert_eq!(s.span(&text, 0, SpanCondition::Simple), 3);
    assert_eq!(s.span(&text, 3, SpanCondition::NotContained), 8);
    assert_eq!(s.span_back(&text, text.len(), SpanCondition::Simple), 8);
    assert_eq!(s.span_back(&text, 8, SpanCondition::NotContained), 3);
    let mut e = UnicodeSet::new();
    e.add(0);
    e.complement();
    assert_eq!(e.ranges(), &[(1, 0x10ffff)]);
    e.complement();
    assert_eq!(e.ranges(), &[(0, 0)]);
    e.clear();
    assert!(e.is_empty());
    assert!(e.apply_int_property_value(0x7777, 1).is_err());
    e.apply_int_property_value(0, 7).unwrap(); // a binary property, value 7: empty
    assert!(e.is_empty());
}

#[test]
fn unescape() {
    let u = |s: &str| unescape_at(&utf16::units(s), 0);
    assert_eq!(u("u0041"), Some((0x41, 5)));
    assert_eq!(u("x{1F600}"), Some((0x1f600, 8)));
    assert_eq!(u("x4"), Some((4, 2)));
    assert_eq!(u("101"), Some((0o101, 3)));
    assert_eq!(u("q"), Some(('q' as i32, 1)));
    assert_eq!(u("z"), Some(('z' as i32, 1)));
    assert_eq!(u("\u{10400}"), Some((0x10400, 2)));
    assert_eq!(u("c\u{10400}"), Some((0, 3)));
    assert_eq!(u("uD801\\x{DC00}"), Some((0x10400, 13)));
    assert_eq!(u("uD801\\q"), Some((0xd801, 5)));
    assert_eq!(u("u12"), None);
    assert_eq!(unescape_at(&[], 0), None);
    assert_eq!(trim_white_space(" 12\t"), "12");
}
