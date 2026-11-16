#![cfg(test)]
use super::*;

/// UTF-16 units of `s`, reading `\n`, `\r`, `\t`, `\\` and `\uXXXX`
/// (a lone surrogate too).
fn units(s: &str) -> Vec<u16> {
    let mut out = Vec::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\\' {
            let mut b = [0; 2];
            out.extend_from_slice(c.encode_utf16(&mut b));
            continue;
        }
        match it.next() {
            Some('n') => out.push(0x0A),
            Some('r') => out.push(0x0D),
            Some('t') => out.push(0x09),
            Some('u') => {
                let hex: String = (0..4).filter_map(|_| it.next()).collect();
                out.push(u16::from_str_radix(&hex, 16).unwrap());
            }
            Some(other) => out.extend([0x5C, other as u16]),
            None => out.push(0x5C),
        }
    }
    out
}

/// Every `find()` of `p` over `text`, as `Matcher` reports them, or the
/// error.
fn finds(p: &str, text: &[u16]) -> String {
    let prog = match compile(p) {
        Ok(prog) => prog,
        Err(e) if super::super::java_regex::is_unsupported(&e) => return "UNSUPPORTED".into(),
        Err(e) => return format!("ERR {e}"),
    };
    let mut out = String::new();
    let (mut from, mut old_last) = (0, 0);
    while from <= text.len() {
        let mut st = State::new(text, prog.group_total(), old_last);
        if !prog.search(from, &mut st).unwrap() {
            break;
        }
        let g = &st.groups;
        out.push_str(&format!("({},{}", g[0], g[1]));
        for c in g[2..].chunks(2) {
            out.push_str(&format!(" {}:{}", c[0], c[1]));
        }
        out.push(')');
        let (s, e) = (g[0] as usize, g[1] as usize);
        from = if s == e { e + 1 } else { e };
        old_last = e;
    }
    out
}

/// Escapes, classes, properties, flags, group syntax and the rarer
/// matcher paths, against `Pattern` (JDK 25; the answers do not differ
/// under JDK 21). `ERR x`: Java's `PatternSyntaxException` description.
#[test]
fn constructs_match_java() {
    let cases: &[(&str, &str, &str)] = &[
        (r"(?U)\w+\W", r"aé1_ -", r"(0,5)"),
        (r"\h+\H\v\V", r"a \t\nb", r""),
        (r"\pL+\PL", r"ab1", r"(0,3)"),
        (r"\Aa", r"aa", r"(0,1)"),
        (r"\b{gx}", r"a", r"ERR Illegal/unsupported escape sequence"),
        (r"\b{g}", r"a", r"UNSUPPORTED"),
        (
            r"(a)\ka",
            r"aa",
            r"ERR \k is not followed by '<' for named capturing group",
        ),
        (
            r"(a)\k<b>",
            r"aa",
            r"ERR named capturing group <b> does not exist",
        ),
        (r"\a\e\f", r"\u0007\u001B\u000C", r"(0,3)"),
        (r"\c", r"a", r"ERR Illegal control escape sequence"),
        (r"\cJ", r"\n", r"(0,1)"),
        (
            r"(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)\11",
            r"abcdefghijkk",
            r"(0,12 0:1 1:2 2:3 3:4 4:5 5:6 6:7 7:8 8:9 9:10 10:11)",
        ),
        (
            r"(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)\111",
            r"abcdefghijkk1",
            r"(0,13 0:1 1:2 2:3 3:4 4:5 5:6 6:7 7:8 8:9 9:10 10:11)",
        ),
        (r"\07\018\0400", r"\u0007\u00018 0", r"(0,5)"),
        (r"\x41\x{1F600}\x{41}", r"A😀A", r"(0,4)"),
        (r"\x{110000}", r"a", r"ERR Hexadecimal codepoint is too big"),
        (r"\x{41", r"a", r"ERR Unclosed hexadecimal escape sequence"),
        (r"\xG", r"a", r"ERR Illegal hexadecimal escape sequence"),
        (r"\uD83D\uDE00\uD83Dx", r"😀\uD83Dx", r"(0,4)"),
        (r"\uD83D\u0041", r"\uD83DA", r"(0,2)"),
        (
            r"(?<1a>x)",
            r"x",
            r"ERR capturing group name does not start with a Latin letter",
        ),
        (
            r"(?<a-b>x)",
            r"x",
            r"ERR named capturing group is missing trailing '>'",
        ),
        (
            r"(?<a>x)(?<a>y)",
            r"xy",
            r"ERR Named capturing group <a> is already defined",
        ),
        (r"(?<=a+)b", r"ab", r"(1,2)"),
        (r"(?$a)", r"a", r"ERR Unknown group type"),
        (r"(?@a)", r"a", r"ERR Unknown group type"),
        (r"(?i)a(?-i)a", r"AaAA", r"(0,2)"),
        (r"(?i-i)a(?is-m:.)", r"a\n", r"(0,2)"),
        (r"(?q)a", r"a", r"ERR Unknown inline modifier"),
        (r"[a[b][c]d]+", r"abcdx", r"(0,4)"),
        (r"[a-c&&b-d&&[c]]", r"abc", r"(2,3)"),
        (r"[[a]&&[b]]", r"ab", r""),
        (r"[&&ab]", r"ab", r"(0,1)(1,2)"),
        (r"[a&&]", r"a&", r"(0,1)"),
        (r"[a&&&b]", r"a&b", r"(0,1)(1,2)(2,3)"),
        (r"[ab&&", r"a", r"ERR Unclosed character class"),
        (r"[^[a]]", r"ab", r"(1,2)"),
        (r"[a&&[^b]c]", r"abc", r"(0,1)"),
        (r"[\pL\p{N}]+", r"a1-", r"(0,2)"),
        (r"[\b]", r"b", r"ERR Illegal/unsupported escape sequence"),
        (r"[\G]", r"G", r"ERR Illegal/unsupported escape sequence"),
        (r"[a-\d]", r"a", r"ERR Illegal character range"),
        (r"[a-\x{63}]+", r"abcd", r"(0,3)"),
        (r"[\x{1F600}-\x{1F601}]+", r"😀😁", r"(0,4)"),
        (r"\p{L", r"a", r"ERR Unclosed character family"),
        (r"\p{}", r"a", r"ERR Empty character family"),
        (r"\p{gc=Lu}\p{general_category=Ll}\p{IsL}", r"Abc", r"(0,3)"),
        (
            r"\p{foo=bar}",
            r"a",
            r"ERR Unknown Unicode property {name=<foo>, value=<bar>}",
        ),
        (
            r"\p{blk=Basic_Latin}\p{block=Greek}\p{sc=Grek}\p{script=Latin}",
            r"aαβb",
            r"(0,4)",
        ),
        (r"(?i)\p{IsLowercase}(?i)\p{javaLowerCase}", r"AB", r"(0,2)"),
        (r"(?iU)\p{Lower}\p{Upper}", r"Ab", r"(0,2)"),
        (
            r"\p{javaFoo}",
            r"a",
            r"ERR Unknown character property name {javaFoo}",
        ),
        (
            r"\p{IsFoo}",
            r"a",
            r"ERR Unknown character property name {IsFoo}",
        ),
        (r"(?U)\p{Alpha}\p{Punct}\p{XDigit}", r"é!f", r"(0,3)"),
        (r"a)", r"a", r"ERR Unmatched closing ')'"),
        (r"(a)\2", r"aa", r""),
        (r"\2a|a", r"a", r"(0,1)"),
        (r"(?<=😀|ab)c", r"😀c abc", r"(2,3)(6,7)"),
        (r"(?<=😀*😀*)b", r"😀b", r"(2,3)"),
        (r"(?<=😀*a*a)b", r"ab", r"(1,2)"),
        (r"(?<=a😀?)b", r"ab a😀b", r"(1,2)(6,7)"),
        (r"(?<!😀{1,2})b", r"😀😀b b", r"(6,7)"),
        (r"(?i)(ab)\1", r"abAB abA", r"(0,4 0:2)"),
        (r"(?iu)(é)\1", r"éÉ éx", r"(0,2 0:1)"),
        (r"(?i)(a)\1", r"aA ab", r"(0,2 0:1)"),
        (r"(?i)(A)\1", r"Aa", r"(0,2 0:1)"),
        (r"(a|ab){0}c", r"abc", r"(2,3 -1:-1)"),
        (r"(a|ab){0,0}?c", r"abc", r"(2,3 -1:-1)"),
        (r"(?:(a)|b)*+\1", r"aab", r""),
        (r"(?x) ( a ) \1 # c", r"aa", r"(0,2 0:1)"),
        (r"(?x)[ a b ]", r" a", r"(1,2)"),
        (r"(?d)a$", r"a\r", r""),
        (r"(?d)(?m)^a", r"\ra", r""),
        (r"(?m)$", r"a\r\nb\n", r"(1,1)(4,4)(5,5)"),
        (r"\R\R", r"\r\n\n", r"(0,3)"),
        (r"(?s).\Z", r"a\n", r"(0,1)(1,2)"),
        (r"\z|\G", r"ab", r"(0,0)(2,2)"),
        (r"(?U)\b.\B", r"é a", r""),
        (r"(?=a)*b", r"b", r"(0,1)"),
        (r"(?<=a){2}b", r"ab", r"(1,2)"),
        (r"(?>a)?b", r"ab", r"(0,2)"),
        // `X{0,1}` is `X?`: a zero-width iteration records its capture.
        (r"(\b){0,1}", r"ab", r"(0,0 0:0)(1,1 -1:-1)(2,2 2:2)"),
        (r"(\B){0,1}(a)", r"aab", r"(0,1 -1:-1 0:1)(1,2 1:1 1:2)"),
        (r"(?=(a)){0,1}?\1", r"ab", r"(0,1 0:1)"),
        (r"a\", r"a", r"ERR Unescaped trailing backslash"),
        // A case-insensitive backreference compares code points until
        // it has covered the group's units.
        (r"(?i)(😀)\1", r"😀😀x", r"(0,4 0:2)"),
        (r"(?i)(😀)\1", r"😀", r""),
        (r"(?iu)(a😀)\1", r"a😀A😀", r"(0,6 0:3)"),
        (r"(?i)(😀a)\1", r"😀a😀A", r"(0,6 0:3)"),
        (r"(?iu)(𐐨)\1", r"𐐨𐐀x", r"(0,4 0:2)"),
        (r"(?iu)(ſ)\1", r"ſs", r"(0,2 0:1)"),
    ];
    for &(p, text, want) in cases {
        let got = finds(p, &units(text));
        match want.strip_prefix("ERR ") {
            Some(why) => assert!(got.starts_with("ERR ") && got.contains(why), "{p:?}: {got}"),
            None => assert_eq!(got, want, "{p:?} on {text:?}"),
        }
    }
}

/// A match whose backtracking overflows the caller's budget inside a
/// negated lookaround or a zero-count quantifier is retried on the deep
/// stack, not taken as the (wrong) match the overflow left behind.
#[test]
fn overflow_under_negation_retries() {
    let text = units(&format!("{}c", "ab".repeat(800)));
    for (p, want) in [
        (r"(?!(?:a|b)*c)", "(1601,1601)"),
        (r"(?>(?:a|b)*c)?", "(0,1601)(1601,1601)"),
        (r"((?:a|b)*c)?+", "(0,1601 0:1601)(1601,1601 -1:-1)"),
    ] {
        assert_eq!(finds(p, &text), want, "{p}");
    }
}

/// The caller's stack is never what a match or a parse is measured
/// against: on a 1 MiB thread with most of it used, a deep match and a
/// pattern nested past [`SHALLOW_NESTING`] still finish (the match on
/// the heap, the parse and its nested sub-programs on a thread of their
/// own).
#[test]
fn small_stack_caller() {
    fn burn(n: usize, f: &dyn Fn()) {
        let buf = [0u8; 1024];
        std::hint::black_box(&buf);
        if n == 0 {
            f()
        } else {
            burn(n - 1, f)
        }
    }
    let h = std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            // In a release build these frames are some 1 KiB each: most
            // of the 1 MiB is gone before the regex runs.
            burn(700, &|| {
                let text = units(&format!("{}c", "ab".repeat(20_000)));
                let prog = compile("(?=a)(?:a|b)*c|(?!(?:a|b)*c)").unwrap();
                let mut st = State::new(&text, prog.group_total(), 0);
                assert!(prog.search(0, &mut st).unwrap());
                assert_eq!(st.group(0), (0, 40_001));
                let nested = format!("{}a{}", "(?=".repeat(40), ")".repeat(40));
                let prog = compile(&nested).unwrap();
                let mut st = State::new(&text, prog.group_total(), 0);
                assert!(prog.search(0, &mut st).unwrap());
                assert_eq!(st.group(0), (0, 0));
                let mut st = State::new(&text[..1], prog.group_total(), 0);
                assert!(!prog.matches_all(&mut st).unwrap());
            });
        })
        .unwrap();
    assert!(h.join().is_ok());
    // What stays on the caller's stack -- a parse 32 levels deep and the
    // passes over its tree, a match within `STACK_BUDGET` -- fits in a
    // 192 KiB thread, a debug build's frames included.
    let h = std::thread::Builder::new()
        .stack_size(192 << 10)
        .spawn(|| {
            for (open, close) in [("(?=", ")"), ("(?>a|", ")+"), ("(", ")"), ("[", "]")] {
                let p = format!(
                    "{}b{}",
                    open.repeat(SHALLOW_NESTING),
                    close.repeat(SHALLOW_NESTING)
                );
                let prog = compile(&p).unwrap();
                let text = units("bb");
                let mut st = State::new(&text, prog.group_total(), 0);
                assert!(prog.search(0, &mut st).unwrap(), "{p}");
            }
        })
        .unwrap();
    assert!(h.join().is_ok());
    // `\b` reads an ASCII unit by the ASCII rule, `(?U)` or not.
    assert!((0..128).all(|c| is_word(c, true) == is_word(c, false)));
}

/// A pattern nested as deep as [`MAX_NESTING`] allows -- one per kind of
/// sub-program a match runs nested (lookahead, negative lookahead, an
/// atomic group, a quantifier over one, a group loop, lookbehind) and a
/// capturing alternation -- compiles or fails with Java's
/// `PatternSyntaxException`, and its match ends or fails as Java's
/// `StackOverflowError`: never a Rust stack overflow, which aborts the
/// process. Run in a child process, so an abort is a failure here, not
/// the end of the test binary.
#[test]
fn deep_nesting_never_aborts() {
    const CHILD: &str = "LUCENE_JAVA_REGEX_DEEP_NESTING_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let path = module_path!().split_once("::").map_or("", |(_, p)| p);
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("{path}::deep_nesting_never_aborts"),
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("deep nesting: done"),
            "{}\n{stdout}\n{stderr}",
            out.status
        );
        return;
    }
    let text = units("bx");
    // Kinds whose match on "bx" takes time linear in the nesting, and
    // those (a repetition tries another iteration at "x", through every
    // level below) whose time is its square, matched shallower.
    for (open, close, linear) in [
        ("(?=a|", ")", true),
        ("(?!x|", ")", true),
        ("(?>a|", ")", true),
        ("(?>a|", "){1}", true),
        ("(?<=a|(?=", "))", true),
        ("(a|", ")", true),
        ("[a", "]", true),
        ("(?>a|", ")+", false),
        ("(?:a|b", ")+", false),
        ("(?:a|", ")*", false),
    ] {
        for n in [3_000, MAX_NESTING, MAX_NESTING + 1] {
            let p = format!("{}b{}", open.repeat(n), close.repeat(n));
            let prog = match compile(&p) {
                Ok(prog) => prog,
                Err(e) => {
                    let want = "Stack overflow during pattern compilation";
                    assert_eq!(
                        e.to_string(),
                        format!("illegal argument: PatternSyntaxException: {want}")
                    );
                    println!("{open} {n}: {e}");
                    continue;
                }
            };
            assert!(n <= MAX_NESTING, "{open} {n}");
            if !linear && n > 3_000 {
                continue;
            }
            let mut st = State::new(&text, prog.group_total(), 0);
            let found = prog.search(0, &mut st);
            let mut st = State::new(&text, prog.group_total(), 0);
            let whole = prog.matches_all(&mut st);
            for r in [&found, &whole] {
                if let Err(e) = r {
                    assert!(e.to_string().contains("StackOverflowError"), "{e}");
                }
            }
            println!("{open} {n}: find {found:?}, matches {whole:?}");
        }
    }
    println!("deep nesting: done");
}

/// Java's `StackOverflowError`: a typed error, from `find()` and
/// `matches()` alike.
#[test]
fn deep_backtracking_overflows() {
    // Five entries an iteration: past `MAX_ENTRIES` at some 210,000.
    let prog = compile("(a|b)*c").unwrap();
    let text = units(&"a".repeat(300_000));
    let mut st = State::new(&text, prog.group_total(), 0);
    let e = prog.search(0, &mut st).unwrap_err();
    assert!(e.to_string().contains("StackOverflowError"), "{e}");
    // ... which is far deeper than Java's own limit (some 1,500 at
    // 1 MiB): 150,000 iterations still match.
    let mut st = State::new(&text[..150_000], prog.group_total(), 0);
    assert!(!prog.search(0, &mut st).unwrap());
    let ok = units(&format!("{}c", "ab".repeat(75_000)));
    let mut st = State::new(&ok, prog.group_total(), 0);
    assert!(prog.matches_all(&mut st).unwrap());
    assert_eq!(st.group(1), (149_999, 150_000));
    // A greedy repetition's back-off positions count too.
    let atomic = compile("(?>a|b)*c").unwrap();
    let long = units(&"a".repeat(1_100_000));
    let mut st = State::new(&long, atomic.group_total(), 0);
    assert!(atomic.search(0, &mut st).is_err());
    let mut st = State::new(&text, prog.group_total(), 0);
    assert!(prog.matches_all(&mut st).is_err());
    let mut st = State::new(&text[..10], prog.group_total(), 0);
    assert!(!prog.matches_all(&mut st).unwrap());
    assert_eq!(st.group(5), (-1, -1));
    // What Java's `compile` makes of its parser's overflow.
    let deep = format!("{}a{}", "(".repeat(100_000), ")".repeat(100_000));
    let want = "Stack overflow during pattern compilation";
    assert!(compile(&deep).unwrap_err().to_string().contains(want));
    let deep = format!("{}a", "[".repeat(100_000));
    assert!(compile(&deep).unwrap_err().to_string().contains(want));
    let named = compile("(?<word>a)").unwrap();
    assert_eq!(
        (named.group_named("word"), named.group_named("x")),
        (Some(1), None)
    );
}
