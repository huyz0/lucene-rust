#![cfg(test)]
use super::*;
use crate::api::test_support::{call, elems, i, s, t, text, vm_call};

fn pat(p: &str, flags: i32) -> Value {
    compile_pattern(p, flags).expect("a pattern")
}

fn matcher(p: &str, input: &str) -> Value {
    call(&pat(p, 0), "matcher", &[s(input)]).unwrap()
}

#[test]
fn operators_find_and_match_either_way_round() {
    let p = pat("b+", 0);
    assert_eq!(
        text(find_op(&mut vm_unused(), &[s("abbc"), p.clone()])),
        "true"
    );
    assert_eq!(
        text(find_op(&mut vm_unused(), &[p.clone(), s("xyz")])),
        "false"
    );
    assert_eq!(
        text(match_op(&mut vm_unused(), &[s("bbb"), p.clone()])),
        "true"
    );
    assert_eq!(
        text(match_op(&mut vm_unused(), &[s("abbc"), p.clone()])),
        "false"
    );
    assert!(text(find_op(&mut vm_unused(), &[Value::Null, p.clone()])).contains("NullPointer"));
    assert!(
        text(find_op(&mut vm_unused(), &[s("x"), s("y")])).contains("ClassCast")
            || text(find_op(&mut vm_unused(), &[s("x"), s("y")])).contains("Pattern")
    );
}

fn vm_unused() -> crate::vm::Vm<'static> {
    struct Env;
    impl crate::vm::Env for Env {
        fn internal(&mut self, _n: &str, _a: &[Value]) -> Result<Value, Throw> {
            unreachable!()
        }
    }
    let env: &'static mut Env = Box::leak(Box::new(Env));
    crate::vm::Vm::new(Rc::new(crate::vm::Program::default()), env)
}

#[test]
fn flags_change_how_a_pattern_reads() {
    let ci = pat("hello", CASE_INSENSITIVE);
    assert_eq!(
        text(find_op(&mut vm_unused(), &[s("HeLLo"), ci.clone()])),
        "true"
    );
    assert_eq!(t(&ci, "flags", &[]), "2");
    let lit = pat("a.c", LITERAL);
    assert_eq!(
        text(find_op(&mut vm_unused(), &[s("abc"), lit.clone()])),
        "false"
    );
    assert_eq!(text(find_op(&mut vm_unused(), &[s("a.c"), lit])), "true");
    let lit_ci = pat("a.c", LITERAL | CASE_INSENSITIVE);
    assert_eq!(text(find_op(&mut vm_unused(), &[s("A.C"), lit_ci])), "true");
    let dot = pat("a.b", DOTALL);
    assert_eq!(text(find_op(&mut vm_unused(), &[s("a\nb"), dot])), "true");
    let nodot = pat("a.b", 0);
    assert_eq!(
        text(find_op(&mut vm_unused(), &[s("a\nb"), nodot])),
        "false"
    );
    assert!(compile_pattern("a", 4).is_err());
    assert!(compile_pattern("(", 0).is_err());
    assert_eq!(t(&pat("x*", 0), "pattern", &[]), "x*");
    assert_eq!(t(&pat("x*", 0), "toString", &[]), "x*");
    assert_eq!(
        text(vm_call(static_lookup("quote", 1).unwrap(), &[s("a.b")])),
        "\\Qa.b\\E"
    );
    assert!(static_lookup("nope", 1).is_none());
}

#[test]
fn matcher_walks_matches_and_groups() {
    let m = matcher("(\\d+)-(?<w>[a-z]+)", "12-ab and 7-cd");
    assert_eq!(t(&m, "groupCount", &[]), "2");
    assert_eq!(t(&m, "find", &[]), "true");
    assert_eq!(t(&m, "group", &[]), "12-ab");
    assert_eq!(t(&m, "group", &[i(1)]), "12");
    assert_eq!(t(&m, "group", &[s("w")]), "ab");
    assert_eq!(t(&m, "namedGroup", &[s("w")]), "ab");
    assert_eq!(t(&m, "start", &[]), "0");
    assert_eq!(t(&m, "end", &[]), "5");
    assert_eq!(t(&m, "start", &[i(2)]), "3");
    assert_eq!(t(&m, "end", &[s("w")]), "5");
    assert_eq!(t(&m, "find", &[]), "true");
    assert_eq!(t(&m, "group", &[i(1)]), "7");
    assert_eq!(t(&m, "start", &[]), "10");
    assert_eq!(t(&m, "find", &[]), "false");
    assert!(t(&m, "group", &[]).contains("IllegalState"));
    assert!(t(&m, "start", &[]).contains("No match available"));
    assert!(t(&m, "group", &[s("nope")]).contains("No group with name <nope>"));
    assert_eq!(t(&m, "reset", &[]), "null");
    assert_eq!(t(&m, "find", &[]), "true");
    assert_eq!(t(&m, "group", &[]), "12-ab");
}

#[test]
fn matcher_modes_and_replacement() {
    assert_eq!(t(&matcher("ab", "abc"), "lookingAt", &[]), "true");
    assert_eq!(t(&matcher("bc", "abc"), "lookingAt", &[]), "false");
    assert_eq!(t(&matcher("abc", "abc"), "matches", &[]), "true");
    assert_eq!(t(&matcher("ab", "abc"), "matches", &[]), "false");
    let m = matcher("o", "foo boo");
    assert_eq!(t(&m, "replaceAll", &[s("0")]), "f00 b00");
    assert_eq!(t(&m, "replaceFirst", &[s("0")]), "f0o boo");
    assert_eq!(
        t(&matcher("(a)(b)", "ab"), "replaceAll", &[s("$2$1")]),
        "ba"
    );
    assert_eq!(t(&matcher("a", "a"), "replaceAll", &[s("\\$")]), "$");
    assert!(t(&matcher("a", "a"), "replaceAll", &[s("$x")]).contains("Illegal group reference"));
    assert!(t(&matcher("a", "a"), "replaceAll", &[s("\\")])
        .contains("character to be escaped is missing"));
    assert_eq!(t(&matcher("a", "a"), "hitEnd", &[]), "false");
    assert_eq!(t(&matcher("x", "ab"), "find", &[i(1)]), "false");
    assert_eq!(t(&matcher("b", "ab"), "find", &[i(1)]), "true");
}

#[test]
fn pattern_split_follows_java() {
    let p = pat(",", 0);
    assert_eq!(elems(call(&p, "split", &[s("a,b,,c,,")])), "[a, b, , c]");
    assert_eq!(elems(call(&p, "split", &[s("a,b,c"), i(2)])), "[a, b,c]");
    assert_eq!(elems(call(&p, "split", &[s("a,b,,"), i(-1)])), "[a, b, , ]");
    assert_eq!(elems(call(&p, "split", &[s("")])), "[]");
    assert_eq!(elems(call(&pat("", 0), "split", &[s("abc")])), "[a, b, c]");
    assert!(t(&p, "asPredicate", &[]).contains("asPredicate"));
    let m = call(&p, "matcher", &[i(1_000), s("a,b")]).unwrap();
    assert_eq!(t(&m, "find", &[]), "true");
}

#[test]
fn translated_syntax_reaches_the_regex_crate() {
    assert_eq!(t(&matcher("\\p{Alpha}+", "abc1"), "find", &[]), "true");
    assert_eq!(t(&matcher("a{2}", "aa"), "matches", &[]), "true");
    assert_eq!(t(&matcher("[a-c&&[^b]]+", "acac"), "matches", &[]), "true");
    assert!(compile_pattern("(a)\\1", 0).is_err());
    assert!(compile_pattern("(?<=a)b", 0).is_err());
}
