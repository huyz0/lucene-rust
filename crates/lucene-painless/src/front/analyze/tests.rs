use std::rc::Rc;

use super::{analyze, Refusal};
use crate::compile::compile;
use crate::front::{parse, Limits};
use crate::jmap::{JMap, Order};
use crate::value::{MapKind, Value};
use crate::vm::{ParamsEnv, Vm};

/// What a lowered script returns.
fn eval(src: &str) -> Value {
    let ast = parse(src, &Limits::default()).expect("parses");
    let script = analyze(&ast).unwrap_or_else(|r| panic!("{src}: {r}"));
    let prog = Rc::new(compile(&script).expect("compiles"));
    let mut env = ParamsEnv {
        params: Value::map(MapKind::HashMap, JMap::new(Order::Hash)),
    };
    Vm::new(prog, &mut env)
        .execute(&[])
        .unwrap_or_else(|_| panic!("{src}: throws"))
}

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
        ("for (def x : [1]) { }", "for-each"),
        ("try { } catch (Exception e) { }", "try"),
        ("int[] a = new int[1];", "array"),
        ("return params", "variable"),
        ("params.x", "variable"),
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

#[test]
fn a_method_of_a_string_is_called_on_its_value() {
    assert!(matches!(eval("'abc'.length()"), Value::Int(3)));
    assert!(matches!(eval("String s = 'hello'; s.substring(1, 3)"), Value::Str(s) if &*s == "el"));
    assert!(matches!(eval("'a'.concat('b').length()"), Value::Int(2)));
}

#[test]
fn a_static_method_and_field_are_named_by_their_class() {
    assert!(matches!(eval("Math.max(1, 2)"), Value::Double(d) if d == 2.0));
    assert!(matches!(eval("Integer.MAX_VALUE"), Value::Int(i32::MAX)));
    assert!(matches!(eval("Integer.parseInt('7')"), Value::Int(7)));
}

#[test]
fn a_def_receiver_is_called_by_name_at_run_time() {
    assert!(matches!(eval("def x = 'abc'; x.length()"), Value::Int(3)));
    assert!(matches!(
        eval("def x = 'abc'; x.substring(1).length()"),
        Value::Int(2)
    ));
}

#[test]
fn a_primitive_receiver_is_boxed() {
    assert!(matches!(eval("int x = 5; x.toString()"), Value::Str(s) if &*s == "5"));
    assert!(matches!(eval("int x = 5; x.compareTo(7)"), Value::Int(-1)));
}

#[test]
fn a_boxed_result_unboxes_where_a_primitive_is_wanted() {
    assert!(matches!(eval("Integer.parseInt('12') + 1"), Value::Int(13)));
    assert!(matches!(
        eval("int x = Integer.valueOf(4); x"),
        Value::Int(4)
    ));
    assert!(matches!(eval("Integer i = 4; i + 1L"), Value::Long(5)));
}

#[test]
fn a_null_safe_call_yields_null_on_a_null_receiver() {
    assert!(matches!(eval("String a = null; a?.length()"), Value::Null));
    assert!(matches!(
        eval("String a = 'ab'; a?.length()"),
        Value::Int(2)
    ));
}

#[test]
fn an_unknown_member_is_a_compile_error() {
    assert!(message("'abc'.nope()").contains("Unknown call [nope] with [0] arguments"));
    assert!(message("'abc'.length(1)").contains("Unknown call [length] with [1] arguments"));
    assert!(message("Integer.NOPE").contains("Unknown field [NOPE]"));
    assert!(message("'abc'.substring('x')").contains("cannot cast"));
}

#[test]
fn a_wrapper_compares_as_an_object_and_widens_as_an_argument() {
    assert!(matches!(eval("Integer a = 5; a == 5"), Value::Bool(true)));
    assert!(matches!(eval("Integer a = 5; a != 6"), Value::Bool(true)));
    assert!(matches!(
        eval("Long b = Long.valueOf(1L); byte u = 1; b.compareTo(u)"),
        Value::Int(0)
    ));
    assert!(matches!(
        refusal("Integer a = 5; true ? a : 6"),
        Refusal::Unsupported(w) if w.contains("conditional")
    ));
}

#[test]
fn a_getter_reads_as_a_field_and_def_fields_are_read_by_name() {
    assert!(matches!(eval("'abc'.empty"), Value::Bool(false)));
    assert!(matches!(eval("def x = ''; x.empty"), Value::Bool(true)));
    assert!(matches!(eval("def x = null; x?.empty"), Value::Null));
    assert!(matches!(
        eval("java.lang.Integer.MIN_VALUE"),
        Value::Int(i32::MIN)
    ));
}

#[test]
fn boxing_conversions_follow_what_widens_and_what_is_cast() {
    assert!(matches!(eval("(Integer)5L"), Value::Int(5)));
    assert!(matches!(eval("Long a = 5; a"), Value::Long(5)));
    assert!(matches!(eval("Integer i = 7; double d = i; d"), Value::Double(d) if d == 7.0));
    assert!(message("long x = 1; Integer i = x;").contains("cannot cast from [long] to [Integer]"));
    assert!(message("Integer i = 1; boolean b = i;").contains("cannot cast from [Integer]"));
    assert!(message("String s = 5;").contains("cannot cast"));
    assert!(message("Integer i = 'a';").contains("cannot cast"));
    for src in [
        "Object o = 5; int x = (int)o;",
        "Number n = 1; short s = (short)n;",
    ] {
        assert!(
            matches!(refusal(src), Refusal::Unsupported(w) if w.contains("cast of")),
            "{src}"
        );
    }
}

#[test]
fn a_call_on_nothing_callable_is_a_compile_error() {
    assert!(message("null.length()").contains("cannot call"));
    assert!(message("'a'.valueOf(1)").contains("Unknown call [valueOf]"));
    assert!(message("String.length()").contains("Unknown call [length]"));
}

#[test]
fn shifts_promote_the_value_and_the_distance_on_their_own() {
    assert!(matches!(eval("1 << 4"), Value::Int(16)));
    assert!(matches!(eval("-16 >> 2"), Value::Int(-4)));
    assert!(matches!(eval("-16 >>> 28"), Value::Int(15)));
    assert!(matches!(eval("1L << 40"), Value::Long(1_099_511_627_776)));
    assert!(matches!(eval("1 << 33L"), Value::Int(2)));
    assert!(matches!(eval("def x = 3; x << 2"), Value::Int(12)));
    assert!(matches!(eval("int x = 1; x <<= 3; x"), Value::Int(8)));
    assert!(message("1.5 << 1").contains("cannot shift"));
    assert!(message("1 << true").contains("cannot shift"));
}

#[test]
fn a_postfix_increment_yields_the_old_value() {
    assert!(matches!(
        eval("int i = 5; int j = i++; j * 10 + i"),
        Value::Int(56)
    ));
    assert!(matches!(
        eval("int i = 5; int j = i--; j * 10 + i"),
        Value::Int(54)
    ));
    assert!(matches!(eval("long i = 5; i++"), Value::Long(5)));
    assert!(matches!(
        eval("def i = 5; def j = i++; j + i"),
        Value::Int(11)
    ));
    assert!(matches!(eval("int i = 5; ++i + i++"), Value::Int(12)));
}
