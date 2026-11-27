use super::{analyze, Refusal};
use crate::front::{parse, Limits};

fn refusal(src: &str) -> Refusal {
    let ast = parse(src, &Limits::default()).expect("parses");
    analyze(&ast).expect_err("is refused")
}

fn message(src: &str) -> String {
    match refusal(src) {
        Refusal::Compile { message, .. } => message,
        Refusal::Unsupported(what) => panic!("unsupported: {what}"),
    }
}

#[test]
fn a_variable_cannot_be_declared_twice_even_in_an_inner_scope() {
    assert!(message("int x = 1; int x = 2;").contains("already defined"));
    assert!(message("int x = 1; if (true) { int x = 2; }").contains("already defined"));
}

#[test]
fn narrowing_needs_a_cast() {
    assert!(message("long a = 1; int b = a;").contains("cannot cast"));
    assert!(message("int a = 1; byte b = a;").contains("cannot cast"));
    assert!(message("boolean a = true; int b = a;").contains("cannot cast"));
}

#[test]
fn a_constant_that_fits_narrows_and_one_that_does_not_is_refused() {
    let ok = parse(
        "byte b = 127; short s = -32768; char c = 65535; b",
        &Limits::default(),
    );
    assert!(analyze(&ok.expect("parses")).is_ok());
    assert!(message("byte b = 128;").contains("cannot cast"));
}

#[test]
fn break_and_continue_need_a_loop() {
    assert!(message("break;").contains("outside of a loop"));
    assert!(message("if (true) { continue; }").contains("outside of a loop"));
}

#[test]
fn a_valued_function_must_return_on_every_path() {
    assert!(message("int f(int x) { if (x > 0) { return 1; } } f(1)").contains("return"));
    assert!(message("void f() { return 1; } f()").contains("void"));
    assert!(message("int f() { return; } f()").contains("return"));
}

#[test]
fn a_loop_that_never_ends_escapes() {
    let src = "int f() { while (true) { } } 1";
    assert!(analyze(&parse(src, &Limits::default()).expect("parses")).is_ok());
    assert!(message("int f() { while (true) { break; } } 1").contains("return"));
}

#[test]
fn operators_check_their_operand_types() {
    assert!(message("boolean b = true; b + 1; 1").contains("not a statement"));
    assert!(message("boolean b = true; return b * 2").contains("cannot apply"));
    assert!(message("return 1 < true").contains("cannot compare"));
    assert!(message("return !5").contains("cannot cast"));
    assert!(message("return ~1.5").contains("cannot apply"));
    assert!(message("return 1 & 1.5").contains("bitwise"));
}

#[test]
fn literals_must_fit() {
    assert!(message("return 2147483648").contains("invalid int literal"));
    assert!(message("return 9223372036854775808L").contains("invalid long literal"));
    let ok = parse("return -2147483648", &Limits::default()).expect("parses");
    assert!(analyze(&ok).is_ok());
}

#[test]
fn a_call_checks_its_arity_and_functions_are_unique() {
    assert!(message("int f(int x) { return x } f(1, 2)").contains("takes 1"));
    assert!(message("int f() { return 1 } int f() { return 2 } 1").contains("already defined"));
}

#[test]
fn what_is_not_lowered_yet_is_named() {
    for (src, what) in [
        ("[1, 2]", "list"),
        ("def x = 1; x.foo()", "method call"),
        ("for (def x : [1]) { }", "for-each"),
        ("try { } catch (Exception e) { }", "try"),
        ("int[] a = new int[1];", "array"),
        ("return params", "variable"),
        ("params.x", "field access"),
        ("return 1 << 2", "shift"),
    ] {
        match refusal(src) {
            Refusal::Unsupported(w) => assert!(w.contains(what), "{src}: {w}"),
            other => panic!("{src}: {other}"),
        }
    }
}

#[test]
fn a_refusal_names_its_cause() {
    let r = refusal("break;");
    assert!(r.to_string().contains(" at "), "{r}");
    assert!(refusal("return params")
        .to_string()
        .starts_with("unsupported:"));
}
