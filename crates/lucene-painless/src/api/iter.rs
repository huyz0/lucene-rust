//! Iterators: what a `for (x : xs)` loop and `Iterable.iterator()` walk.
//!
//! An iterator snapshots its collection's elements when it is created.

use std::cell::Cell;

use crate::api::npe;
use crate::object::{downcast, JObject};
use crate::types::canonical;
use crate::value::Value;
use crate::vm::{Throw, Vm};

#[derive(Debug)]
pub struct JIter {
    pub items: Vec<Value>,
    pub pos: Cell<usize>,
}

impl JIter {
    pub fn value(items: Vec<Value>) -> Value {
        Value::object(JIter {
            items,
            pos: Cell::new(0),
        })
    }
}

impl JObject for JIter {
    fn class_name(&self) -> &str {
        "java.util.Iterator"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn instance_of(&self, class: &str) -> bool {
        matches!(
            class,
            "java.util.Iterator" | "java.util.ListIterator" | "java.lang.Object"
        )
    }

    fn call(
        &self,
        _vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(match (name, args.len()) {
            ("hasNext", 0) => Ok(Value::Bool(self.pos.get() < self.items.len())),
            ("next", 0) => self.next(),
            _ => return None,
        })
    }
}

impl JIter {
    fn next(&self) -> Result<Value, Throw> {
        let p = self.pos.get();
        match self.items.get(p) {
            Some(v) => {
                self.pos.set(p + 1);
                Ok(v.clone())
            }
            None => Err(Throw::no_message("java.util.NoSuchElementException")),
        }
    }
}

/// The elements an `Iterable` (or array) iterates, in order.
pub fn elements(vm: &mut Vm, v: &Value) -> Result<Vec<Value>, Throw> {
    Ok(match v {
        Value::Null => return Err(npe()),
        Value::List(l) => l.borrow().items.clone(),
        Value::Set(s) => s.borrow().map.keys(),
        Value::Array(a) => a.borrow().items.clone(),
        Value::Object(o) => match o.iterate(vm) {
            Some(r) => r?,
            None => {
                if let Some(it) = o.as_any().downcast_ref::<JIter>() {
                    it.items[it.pos.get()..].to_vec()
                } else {
                    return Err(Throw::illegal_arg(format!(
                        "Cannot iterate over [{}]",
                        canonical(o.class_name())
                    )));
                }
            }
        },
        other => {
            return Err(Throw::illegal_arg(format!(
                "Cannot iterate over [{}]",
                canonical(&other.class_name())
            )))
        }
    })
}

pub fn iterator(vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    Ok(JIter::value(elements(vm, &args[0])?))
}

pub fn has_next(vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    match downcast::<JIter>(&args[0]) {
        Some(it) => Ok(Value::Bool(it.pos.get() < it.items.len())),
        None => crate::api::dynamic::call(vm, &args[0], "hasNext", &[]),
    }
}

pub fn next(vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    match downcast::<JIter>(&args[0]) {
        Some(it) => it.next(),
        None => crate::api::dynamic::call(vm, &args[0], "next", &[]),
    }
}
