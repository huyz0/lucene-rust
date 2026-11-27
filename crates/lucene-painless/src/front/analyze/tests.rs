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
    for (src, what) in [("return doc", "variable")] {
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
    assert!(refusal("return doc")
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

#[test]
fn objects_lists_and_maps_are_built() {
    assert!(matches!(eval("new ArrayList().size()"), Value::Int(0)));
    assert!(matches!(
        eval("new StringBuilder('ab').length()"),
        Value::Int(2)
    ));
    assert!(matches!(eval("[1, 'a', 2.5].size()"), Value::Int(3)));
    assert!(matches!(eval("[1, 2, 3].get(1)"), Value::Int(2)));
    assert!(matches!(eval("[:].size()"), Value::Int(0)));
    assert!(matches!(eval("['a': 1, 'b': 2].get('b')"), Value::Int(2)));
    assert!(matches!(eval("new ArrayList(); 7"), Value::Int(7)));
    assert!(message("new ArrayList(1, 2, 3)").contains("Unknown new call"));
    assert!(message("new List()").contains("Unknown new call"));
}

#[test]
fn reference_equality_compares_as_painless_does() {
    assert!(matches!(
        eval("def a = 1; def b = 1; a === b"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval("def x = (byte)7; def y = (int)7; x === y"),
        Value::Bool(false)
    ));
    assert!(matches!(
        eval("Map m = new HashMap(); m === null"),
        Value::Bool(false)
    ));
    assert!(matches!(
        eval("Map m = new HashMap(); m !== null"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval("boolean x = true; boolean y = true; x === y"),
        Value::Bool(true)
    ));
}

#[test]
fn params_map_keys_and_elvis_read_as_painless_does() {
    assert!(matches!(eval("params.a"), Value::Null));
    assert!(matches!(eval("params.a ?: 5"), Value::Int(5)));
    assert!(matches!(eval("Map m = ['k': 3]; m.k"), Value::Int(3)));
    assert!(matches!(eval("Map m = ['k': 3]; m.nope?.x"), Value::Null));
    assert!(matches!(
        eval("def m = ['k': ['j': 4]]; m.k.j"),
        Value::Int(4)
    ));
    assert!(matches!(eval("String s = null; s ?: 'z'"), Value::Str(z) if &*z == "z"));
    assert!(message("int i = 1; i ?: 2").contains("elvis"));
    assert!(message("'a'.nope").contains("Unknown field [nope]"));
}

#[test]
fn arrays_are_made_indexed_and_assigned() {
    assert!(matches!(
        eval("int[] a = new int[3]; a.length"),
        Value::Int(3)
    ));
    assert!(matches!(
        eval("int[] a = new int[] {4, 5, 6}; a[1] + a[-1]"),
        Value::Int(11)
    ));
    assert!(matches!(
        eval("int[][] a = new int[2][3]; a[1][2] = 7; a[1][2]"),
        Value::Int(7)
    ));
    assert!(matches!(
        eval("int[] a = new int[2]; a[0] += 5; a[0] *= 3; a[0]"),
        Value::Int(15)
    ));
    assert!(matches!(
        eval("int[] a = new int[2]; a[1] = 4; int b = a[1]++; b * 10 + a[1]"),
        Value::Int(45)
    ));
    assert!(matches!(
        eval("int[] a = new int[2]; int b = ++a[0] + a[0]--; b * 10 + a[0]"),
        Value::Int(20)
    ));
    assert!(
        matches!(eval("String[] s = new String[] {'a', 'b'}; s[1]"), Value::Str(x) if &*x == "b")
    );
    assert!(matches!(
        eval("def a = new int[] {1, 2}; a[1]"),
        Value::Int(2)
    ));
    assert!(message("int[] a = new int[1]; a['x']").contains("cannot cast"));
    assert!(message("int a = 1; a[0]").contains("Cannot apply []"));
    assert!(matches!(
        refusal("int[] a = new int[1]; long[] b = a;"),
        Refusal::Unsupported(_)
    ));
}

#[test]
fn elements_and_members_of_lists_maps_and_defs_are_assigned() {
    assert!(matches!(
        eval("List l = [1, 2]; l[0] = 9; l[0] + l[-1]"),
        Value::Int(11)
    ));
    assert!(matches!(
        eval("Map m = new HashMap(); m['k'] = 3; m['k'] += 4; m.k"),
        Value::Int(7)
    ));
    assert!(matches!(
        eval("Map m = new HashMap(); m.k = 3; m['k']"),
        Value::Int(3)
    ));
    assert!(matches!(
        eval("def m = new HashMap(); m.k = 5; m.k++; m.k"),
        Value::Int(6)
    ));
    assert!(matches!(
        eval("def a = [1, 2]; a[0] = 8; a[0]"),
        Value::Int(8)
    ));
    assert!(matches!(
        eval("List l = [1]; int x = (l[0] = 6); x"),
        Value::Int(6)
    ));
    assert!(message("String s = 'a'; s?.x = 1").contains("null-safe"));
}

#[test]
fn instanceof_and_for_each_follow_the_value_and_the_collection() {
    assert!(matches!(
        eval("def x = 'a'; x instanceof String"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval("int x = 1; x instanceof Integer"),
        Value::Bool(true)
    ));
    assert!(matches!(
        eval("def x = []; x instanceof Map"),
        Value::Bool(false)
    ));
    assert!(matches!(
        eval("int s = 0; for (int i : [1, 2, 3]) { s += i } s"),
        Value::Int(6)
    ));
    assert!(matches!(
        eval("int s = 0; for (x in new int[] {4, 5}) { s += x } s"),
        Value::Int(9)
    ));
    assert!(matches!(
        eval("long s = 0; for (long i : new int[] {4, 5}) { s += i } s"),
        Value::Long(9)
    ));
    assert!(message("for (int i : 5) { }").contains("Illegal for each"));
    assert!(message("for (int i : 'ab') { }").contains("Illegal for each"));
}

#[test]
fn try_catch_and_throw_follow_the_exception_type() {
    assert!(matches!(
        eval("int x = 1; try { x = 2 } catch (Exception e) { x = 3 } x"),
        Value::Int(2)
    ));
    assert!(matches!(
        eval("int x = 1; try { throw new RuntimeException('a') } catch (Exception e) { x = 3 } x"),
        Value::Int(3)
    ));
    assert!(message("throw 'a'").contains("cannot cast"));
    assert!(message("try { } catch (String e) { }").contains("not an exception type"));
}

#[test]
fn lambdas_and_method_references_implement_the_parameter_interface() {
    assert!(matches!(
        eval("List l = new ArrayList(); l.add('looooong'); l.add('short'); l.sort((a, b) -> a.length() - b.length()); l.get(0)"),
        Value::Str(s) if &*s == "short"
    ));
    assert!(matches!(
        eval("int foo(Function f) { return f.apply(1) } int k = 4; foo(x -> x + k)"),
        Value::Int(5)
    ));
    assert!(matches!(
        eval("int foo(Function f) { return f.apply(1) } foo(x -> foo(y -> x + 1))"),
        Value::Int(2)
    ));
    assert!(matches!(
        eval("int mycompare(int i, int j) { j - i } List l = new ArrayList(); l.add(1); l.add(2); l.sort(this::mycompare); l.get(0)"),
        Value::Int(2)
    ));
    assert!(matches!(
        eval("List l = new ArrayList(); l.add(3); l.add(1); l.sort(Integer::compare); l.get(0)"),
        Value::Int(1)
    ));
    assert!(matches!(
        eval("def l = new ArrayList(); l.add(1); l.add(2); l.removeIf(x -> x == 1); l.size()"),
        Value::Int(1)
    ));
    assert!(message("List l = new ArrayList(); l.sort(x -> 1)")
        .contains("Incorrect number of parameters"));
    assert!(message("'a'.concat(x -> 1)").contains("not a functional interface"));
    assert!(
        message("int k = 1; List l = new ArrayList(); l.removeIf(k -> true)")
            .contains("already defined")
    );
}

#[test]
fn regex_literals_and_operators_follow_painless() {
    assert!(matches!(eval("'foo' ==~ /fo+/"), Value::Bool(true)));
    assert!(matches!(eval("'a foo b' ==~ /foo/"), Value::Bool(false)));
    assert!(matches!(eval("'a foo b' =~ /foo/"), Value::Bool(true)));
    assert!(matches!(eval("'FOO' ==~ /foo/i"), Value::Bool(true)));
    assert!(matches!(
        eval("Matcher m = /(f)(o)o/.matcher('foo'); m.find(); m.group(2)"),
        Value::Str(s) if &*s == "o"
    ));
    assert!(matches!(eval("/,/.split('a,b').length"), Value::Int(2)));
    assert!(message("'a' ==~ 1").contains("cannot cast"));
}
