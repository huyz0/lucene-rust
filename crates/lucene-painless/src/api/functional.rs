//! Function values built from others: `Comparator.reversed`,
//! `thenComparing`, `Function.andThen`, `Predicate.negate`, and the
//! `Comparator` / `Function` factories.

use std::cmp::Ordering;

use crate::api::int_arg;
use crate::compare;
use crate::lambda::invoke;
use crate::object::JObject;
use crate::value::Value;
use crate::vm::{ApiFn, Throw, Vm};

#[derive(Debug, Clone)]
pub enum Composed {
    /// `c.reversed()`; `None` is `Comparator.reverseOrder()`.
    Reversed(Option<Value>),
    NaturalOrder,
    ThenComparing(Value, Value),
    /// `Comparator.comparing(keyExtractor)`.
    Comparing(Value),
    AndThen(Value, Value),
    Negate(Value),
    And(Value, Value),
    Or(Value, Value),
    Identity,
}

impl Composed {
    pub fn invoke(&self, vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
        match self {
            Composed::Reversed(c) => {
                let o = cmp(vm, c.as_ref(), &args[1], &args[0])?;
                Ok(Value::Int(compare::sign(o)))
            }
            Composed::NaturalOrder => Ok(Value::Int(compare::sign(compare::compare_to(
                &args[0], &args[1],
            )?))),
            Composed::ThenComparing(a, b) => {
                let o = cmp(vm, Some(a), &args[0], &args[1])?;
                if o != Ordering::Equal {
                    return Ok(Value::Int(compare::sign(o)));
                }
                let o = cmp(vm, Some(b), &args[0], &args[1])?;
                Ok(Value::Int(compare::sign(o)))
            }
            Composed::Comparing(key) => {
                let ka = invoke(vm, key, &args[0..1])?;
                let kb = invoke(vm, key, &args[1..2])?;
                Ok(Value::Int(compare::sign(compare::compare_to(&ka, &kb)?)))
            }
            Composed::AndThen(f, g) => {
                let r = invoke(vm, f, args)?;
                invoke(vm, g, &[r])
            }
            Composed::Negate(p) => Ok(Value::Bool(!truthy(&invoke(vm, p, args)?)?)),
            Composed::And(p, q) => Ok(Value::Bool(
                truthy(&invoke(vm, p, args)?)? && truthy(&invoke(vm, q, args)?)?,
            )),
            Composed::Or(p, q) => Ok(Value::Bool(
                truthy(&invoke(vm, p, args)?)? || truthy(&invoke(vm, q, args)?)?,
            )),
            Composed::Identity => Ok(args.first().cloned().unwrap_or_default()),
        }
    }
}

fn truthy(v: &Value) -> Result<bool, Throw> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Null => Err(Throw::no_message("java.lang.NullPointerException")),
        o => Err(Throw::class_cast(format!(
            "{} is not a boolean",
            o.class_name()
        ))),
    }
}

fn cmp(vm: &mut Vm, c: Option<&Value>, a: &Value, b: &Value) -> Result<Ordering, Throw> {
    match c {
        Some(c) => Ok(int_arg(&invoke(vm, c, &[a.clone(), b.clone()])?)?.cmp(&0)),
        None => compare::compare_to(a, b),
    }
}

impl JObject for Composed {
    fn class_name(&self) -> &str {
        match self {
            Composed::Reversed(_)
            | Composed::NaturalOrder
            | Composed::ThenComparing(..)
            | Composed::Comparing(_) => "java.util.Comparator",
            Composed::AndThen(..) | Composed::Identity => "java.util.function.Function",
            _ => "java.util.function.Predicate",
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn instance_of(&self, class: &str) -> bool {
        class == self.class_name() || class == "java.lang.Object"
    }

    fn call(
        &self,
        vm: &mut Vm,
        this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(match (name, args.len()) {
            ("reversed", 0) => Ok(reversed(this.clone())),
            ("thenComparing", 1) => Ok(then_comparing(this.clone(), args[0].clone())),
            ("andThen", 1) => Ok(and_then(this.clone(), args[0].clone())),
            ("negate", 0) => Ok(negate(this.clone())),
            ("and", 1) => Ok(and(this.clone(), args[0].clone())),
            ("or", 1) => Ok(or(this.clone(), args[0].clone())),
            ("equals", 1) | ("hashCode", 0) | ("toString", 0) => return None,
            _ => self.invoke(vm, args),
        })
    }
}

pub fn reversed(c: Value) -> Value {
    Value::object(Composed::Reversed(Some(c)))
}

pub fn reverse_order() -> Value {
    Value::object(Composed::Reversed(None))
}

pub fn then_comparing(a: Value, b: Value) -> Value {
    Value::object(Composed::ThenComparing(a, b))
}

pub fn and_then(f: Value, g: Value) -> Value {
    Value::object(Composed::AndThen(f, g))
}

pub fn negate(p: Value) -> Value {
    Value::object(Composed::Negate(p))
}

pub fn and(p: Value, q: Value) -> Value {
    Value::object(Composed::And(p, q))
}

pub fn or(p: Value, q: Value) -> Value {
    Value::object(Composed::Or(p, q))
}

pub fn comparator_static(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("naturalOrder", 0) => |_vm, _a| Ok(Value::object(Composed::NaturalOrder)),
        ("reverseOrder", 0) => |_vm, _a| Ok(reverse_order()),
        ("comparing", 1) | ("comparingInt", 1) | ("comparingLong", 1) | ("comparingDouble", 1) => {
            |_vm, a| Ok(Value::object(Composed::Comparing(a[0].clone())))
        }
        _ => return None,
    })
}

pub fn function_static(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("identity", 0) => |_vm, _a| Ok(Value::object(Composed::Identity)),
        _ => return None,
    })
}
