#![cfg(test)]
use super::*;
use crate::reader::StrReader;
use crate::util::canned::{render, Canned};

fn run(t: &mut dyn Tokenizer, text: &str) -> String {
    t.set_reader(Box::new(StrReader::new(text))).unwrap();
    render(t)
}

fn p(s: &str) -> JavaPattern {
    JavaPattern::compile(s).unwrap()
}

#[test]
fn pattern_tokenizer() {
    let mut t = PatternTokenizer::new(&p("[ ,]+"), -1).unwrap();
    assert_eq!(run(&mut t, ",a, b😀,"), "a:1:2:1:1 b😀:4:7:1:1|8|0");
    assert_eq!(run(&mut t, "ab"), "ab:0:2:1:1|2|0");
    assert_eq!(run(&mut t, ""), "|0|0");
    let mut t = PatternTokenizer::new(&p("([a-z]+)([0-9]*)"), 1).unwrap();
    assert_eq!(run(&mut t, "ab12 C cd"), "ab:0:2:1:1 cd:7:9:1:1|9|0");
    let mut t = PatternTokenizer::new(&p("(x*)"), 1).unwrap();
    assert_eq!(run(&mut t, "ab"), "|2|0");
    assert!(PatternTokenizer::new(&p("(a)"), 2).is_err());
}

#[test]
fn replace_capture_and_typing() {
    let mut f = PatternReplaceFilter::new(
        Canned::parse("hello:0:5:1:1 xyz:6:9:1:1|9|0"),
        p("[aeiou]"),
        Some("_"),
        true,
    );
    assert_eq!(render(&mut f), "h_ll_:0:5:1:1 xyz:6:9:1:1|9|0");
    let mut f = PatternReplaceFilter::new(Canned::parse("hello:0:5:1:1|5|0"), p("l"), None, false);
    assert_eq!(render(&mut f), "helo:0:5:1:1|5|0");
    let pats = [p("([A-Z][a-z]+)"), p("([0-9]+)")];
    let mut c = Canned::parse("x:0:1:1:1 y:2:3:1:1|3|0");
    c.set_terms(&["FooBar12", "zz"]);
    let mut f = PatternCaptureGroupTokenFilter::new(c, true, &pats);
    assert_eq!(
        render(&mut f),
        "FooBar12:0:1:1:1 Foo:0:1:0:1 Bar:0:1:0:1 12:0:1:0:1 zz:2:3:1:1|3|0"
    );
    let mut c = Canned::parse("x:0:1:1:1|1|0");
    c.set_terms(&["FooBar12"]);
    let mut f = PatternCaptureGroupTokenFilter::new(c, false, &pats);
    assert_eq!(render(&mut f), "Foo:0:1:1:1 Bar:0:1:0:1 12:0:1:0:1|1|0");
    let rules = vec![PatternTypingRule {
        pattern: p("^(\\d+)$"),
        flags: 2,
        type_template: "num_$1".into(),
    }];
    let mut f = PatternTypingFilter::new(Canned::parse("42:0:2:1:1 a:3:4:1:1|4|0"), rules);
    let mut out = Vec::new();
    crate::token_stream::consume(&mut f, |a| {
        out.push((a.token_type().to_string(), a.flags()))
    })
    .unwrap();
    assert_eq!(
        out,
        vec![("num_42".to_string(), 2), ("word".to_string(), 0)]
    );
    let rules = vec![PatternTypingRule {
        pattern: p("^([A-Z])"),
        flags: 1,
        type_template: "c_$1".into(),
    }];
    let mut c = Canned::parse("x:0:1:1:1");
    c.set_terms(&["Hello"]);
    let mut f = PatternTypingFilter::new(c, rules);
    let mut out = Vec::new();
    crate::token_stream::consume(&mut f, |a| out.push(a.token_type().to_string())).unwrap();
    assert_eq!(out, vec!["c_Hello"]);
}

#[test]
fn replace_char_filter() {
    let mut f =
        PatternReplaceCharFilter::new(p("([a-z]+)-([a-z]+)"), "$2_$1x", StrReader::new("ab-cd e"));
    let mut buf = [0u16; 32];
    let n = f.read(&mut buf).unwrap();
    assert_eq!(String::from_utf16(&buf[..n]).unwrap(), "cd_abx e");
    assert_eq!(f.read(&mut buf).unwrap(), 0);
    assert_eq!(
        (
            f.correct_offset(5),
            f.correct_offset(6),
            f.correct_offset(8)
        ),
        (4, 5, 7)
    );
    let mut f = PatternReplaceCharFilter::new(p("abc"), "z", StrReader::new("abc d"));
    let n = f.read(&mut buf).unwrap();
    assert_eq!(String::from_utf16(&buf[..n]).unwrap(), "z d");
    assert_eq!((f.correct_offset(1), f.correct_offset(2)), (3, 4));
}

/// Every match's UTF-16 offsets come from the matcher: rescanning the
/// text before each match made this quadratic (minutes for 200 000
/// non-ASCII characters and 100 000 matches).
#[test]
fn replace_char_filter_is_linear() {
    let text = "é,".repeat(100_000);
    let started = std::time::Instant::now();
    let mut f = PatternReplaceCharFilter::new(p(","), "--", StrReader::new(&text));
    let mut buf = vec![0u16; 400_000];
    let n = f.read(&mut buf).unwrap();
    assert_eq!(n, 300_000);
    assert_eq!(f.correct_offset(299_999), 199_999);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}

/// Java's empty-match rule: after a non-empty match an empty one may
/// follow at its end (`"abxd".replaceAll("x*", "-")` is `-a-b--d-`).
#[test]
fn empty_matches_follow_java() {
    let mut f =
        PatternReplaceFilter::new(Canned::parse("abxd:0:4:1:1|4|0"), p("x*"), Some("-"), true);
    assert_eq!(render(&mut f), "-a-b--d-:0:4:1:1|4|0");
    let mut f = PatternReplaceCharFilter::new(p("x*"), "-", StrReader::new("abxd"));
    let mut buf = [0u16; 32];
    let n = f.read(&mut buf).unwrap();
    assert_eq!(String::from_utf16(&buf[..n]).unwrap(), "-a-b--d-");
    // Between the surrogates of a pair the output keeps each half.
    let mut f = PatternReplaceCharFilter::new(p("x*"), "-", StrReader::new("😀"));
    let n = f.read(&mut buf).unwrap();
    assert_eq!(buf[..n], [0x2D, 0xD83D, 0x2D, 0xDE00, 0x2D]);
    let rules = vec![PatternTypingRule {
        pattern: p("x*"),
        flags: 1,
        type_template: "<$0>".into(),
    }];
    let mut c = Canned::parse("x:0:1:1:1");
    c.set_terms(&["axb"]);
    let mut f = PatternTypingFilter::new(c, rules);
    let mut out = Vec::new();
    crate::token_stream::consume(&mut f, |a| out.push(a.token_type().to_string())).unwrap();
    assert_eq!(out, vec!["<>axb"]);
}

#[test]
fn simple_patterns() {
    let mut t = SimplePatternTokenizer::new("[a-z]+[0-9]*").unwrap();
    assert_eq!(
        run(&mut t, "ab12 Cd x😀y"),
        "ab12:0:4:1:1 d:6:7:1:1 x:8:9:1:1 y:11:12:1:1|12|0"
    );
    let mut t = SimplePatternTokenizer::new("abc").unwrap();
    assert_eq!(run(&mut t, "ababc ab"), "abc:2:5:1:1|8|0");
    let mut t = SimplePatternSplitTokenizer::new("[ ,]+").unwrap();
    assert_eq!(
        run(&mut t, ", a,b ,c😀"),
        "a:2:3:1:1 b:4:5:1:1 c😀:7:10:1:1|10|0"
    );
    let mut t = SimplePatternSplitTokenizer::new("ab").unwrap();
    assert_eq!(run(&mut t, "xaay"), "xaay:0:4:1:1|4|0");
    assert_eq!(run(&mut t, "ab"), "|2|0");
    assert!(SimplePatternTokenizer::new("(").is_err());
    let a = lucene_util::automaton::automata::make_string("ab");
    let b = lucene_util::automaton::automata::make_string("ac");
    let nfa = operations::union(&[&a, &b]);
    assert!(!nfa.is_deterministic());
    assert!(SimplePatternTokenizer::from_automaton(&nfa).is_err());
    assert!(SimplePatternSplitTokenizer::from_automaton(&nfa).is_err());
}
