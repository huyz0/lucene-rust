#![cfg(test)]
use super::*;
use crate::api::test_support::{i, s, text, vm_call};

fn obj(name: &str, args: &[Value]) -> String {
    match objects_lookup(name, args.len()) {
        Some(f) => text(vm_call(f, args)),
        None => format!("no Objects.{name}/{}", args.len()),
    }
}

#[test]
fn objects_helpers() {
    assert_eq!(obj("equals", &[s("a"), s("a")]), "true");
    assert_eq!(obj("equals", &[Value::Null, Value::Null]), "true");
    assert_eq!(obj("equals", &[s("a"), Value::Null]), "false");
    assert_eq!(obj("deepEquals", &[i(1), i(1)]), "true");
    assert_eq!(obj("hashCode", &[s("abc")]), "96354");
    assert_eq!(obj("hashCode", &[Value::Null]), "0");
    let arr = Value::array("def", vec![i(1), i(2)]);
    assert_eq!(obj("hash", &[arr]), "994");
    assert_eq!(obj("hash", &[s("a")]), "128");
    assert_eq!(obj("isNull", &[Value::Null]), "true");
    assert_eq!(obj("nonNull", &[Value::Null]), "false");
    assert_eq!(obj("toString", &[i(5)]), "5");
    assert_eq!(obj("toString", &[Value::Null]), "null");
    assert_eq!(obj("toString", &[Value::Null, s("dflt")]), "dflt");
    assert_eq!(obj("toString", &[i(5), s("dflt")]), "5");
    assert_eq!(obj("requireNonNull", &[s("x")]), "x");
    assert!(obj("requireNonNull", &[Value::Null]).contains("NullPointer"));
    assert!(obj("requireNonNull", &[Value::Null, s("what")]).contains("what"));
    assert_eq!(obj("requireNonNull", &[s("x"), s("what")]), "x");
    assert_eq!(obj("compare", &[s("a"), s("a"), Value::Null]), "0");
}

#[test]
fn system_clock_and_identity() {
    let now = vm_call(system_lookup("currentTimeMillis", 0).unwrap(), &[]).unwrap();
    assert!(matches!(now, Value::Long(ms) if ms > 1_600_000_000_000));
    let ns = vm_call(system_lookup("nanoTime", 0).unwrap(), &[]).unwrap();
    assert!(matches!(ns, Value::Long(n) if n > 0));
    let h = vm_call(system_lookup("identityHashCode", 1).unwrap(), &[s("a")]).unwrap();
    assert!(matches!(h, Value::Int(_)));
    assert!(system_lookup("exit", 1).is_none());
    assert!(objects_lookup("nope", 1).is_none());
}

#[test]
fn object_methods_on_any_receiver() {
    let mut env = NoEnvForTests;
    let mut vm = crate::vm::Vm::new(std::rc::Rc::new(crate::vm::Program::default()), &mut env);
    let call = |vm: &mut Vm, recv: &Value, name: &str, args: &[Value]| {
        object_call(vm, recv, name, args).map(text)
    };
    assert_eq!(call(&mut vm, &s("a"), "equals", &[s("a")]).unwrap(), "true");
    assert_eq!(call(&mut vm, &s("abc"), "hashCode", &[]).unwrap(), "96354");
    assert_eq!(call(&mut vm, &i(3), "toString", &[]).unwrap(), "3");
    assert_eq!(
        call(&mut vm, &s("a"), "compareTo", &[s("b")]).unwrap(),
        "-1"
    );
    assert_eq!(call(&mut vm, &i(2), "compareTo", &[i(1)]).unwrap(), "1");
    assert!(call(&mut vm, &i(2), "other", &[]).is_none());
}

struct NoEnvForTests;

impl crate::vm::Env for NoEnvForTests {
    fn internal(&mut self, _name: &str, _args: &[Value]) -> Result<Value, Throw> {
        unreachable!()
    }
}

#[test]
fn exceptions_a_script_constructs() {
    let e = exception_of("java.lang.IllegalStateException", &[s("boom")], 7).unwrap();
    assert_eq!(e.java_to_string(), "java.lang.IllegalStateException: boom");
    let bare = exception_of("java.lang.RuntimeException", &[], 0).unwrap();
    assert_eq!(bare.java_to_string(), "java.lang.RuntimeException");
    let from_null = exception_of("java.lang.RuntimeException", &[Value::Null], 0).unwrap();
    assert_eq!(from_null.java_to_string(), "java.lang.RuntimeException");
    let from_int = exception_of("java.lang.Exception", &[i(4)], 0).unwrap();
    assert_eq!(from_int.java_to_string(), "java.lang.Exception: 4");
    assert!(exception_of("not.an.Exception", &[], 0).is_err());
}
