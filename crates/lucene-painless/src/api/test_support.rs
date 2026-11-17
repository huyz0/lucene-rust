#![cfg(test)]
//! Shared helpers of the API unit tests: call a method or a static the way
//! compiled code does, and print a result or a throw as one string.

use std::rc::Rc;

use crate::value::Value;
use crate::vm::{Env, Program, Throw, Vm};

struct NoEnv;

impl Env for NoEnv {
    fn internal(&mut self, _name: &str, _args: &[Value]) -> Result<Value, Throw> {
        unreachable!("the API tests use no environment")
    }
}

fn vm_run<R>(f: impl FnOnce(&mut Vm) -> R) -> R {
    let mut env = NoEnv;
    let mut vm = Vm::new(Rc::new(Program::default()), &mut env);
    f(&mut vm)
}

/// `recv.name(args)`.
pub fn call(recv: &Value, name: &str, args: &[Value]) -> Result<Value, Throw> {
    vm_run(|vm| super::dynamic::call(vm, recv, name, args))
}

/// Runs a bare API function (a constructor, say) with `args`.
pub fn vm_call(f: crate::vm::ApiFn, args: &[Value]) -> Result<Value, Throw> {
    vm_run(|vm| f(vm, args))
}

/// `Class.name(args)` through the static lookup of `class`.
pub fn stat(
    lookup: fn(&str, usize) -> Option<crate::vm::ApiFn>,
    name: &str,
    args: &[Value],
) -> Result<Value, Throw> {
    vm_run(|vm| lookup(name, args.len()).expect("a static")(vm, args))
}

/// The result as Java would print it, or `class: message` of the throw.
pub fn text(r: Result<Value, Throw>) -> String {
    r.map(|v| v.java_to_string())
        .unwrap_or_else(|e| format!("{}: {}", e.class(), e.message().unwrap_or_default()))
}

/// A result's elements as `[a, b]` when it is an array, else as [`text`].
pub fn elems(r: Result<Value, Throw>) -> String {
    match r {
        Ok(Value::Array(a)) => format!(
            "[{}]",
            a.borrow()
                .items
                .iter()
                .map(Value::java_to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => text(other),
    }
}

/// `recv.name(args)` as [`text`].
pub fn t(recv: &Value, name: &str, args: &[Value]) -> String {
    text(call(recv, name, args))
}

pub fn s(x: &str) -> Value {
    Value::str(x)
}

pub fn i(x: i32) -> Value {
    Value::Int(x)
}
