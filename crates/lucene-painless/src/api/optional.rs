//! `java.util.Optional`: a value that may be absent.

use std::any::Any;

use crate::lambda::invoke_as;
use crate::object::JObject;
use crate::value::Value;
use crate::vm::{ApiFn, Throw, Vm};

const CLASS: &str = "java.util.Optional";
const SUPPLIER: &str = "java.util.function.Supplier";
const CONSUMER: &str = "java.util.function.Consumer";
const FUNCTION: &str = "java.util.function.Function";
const PREDICATE: &str = "java.util.function.Predicate";

/// An `Optional`: `None` for the empty one.
#[derive(Debug)]
struct Optional(Option<Value>);

fn wrap(v: Option<Value>) -> Value {
    Value::object(Optional(v))
}

/// `Optional.ofNullable`: the empty one for a `null`.
fn nullable(v: Value) -> Value {
    wrap(match v {
        Value::Null => None,
        v => Some(v),
    })
}

/// The static methods of `Optional`.
pub fn static_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("empty", 0) => |_vm, _a| Ok(wrap(None)),
        ("of", 1) => |_vm, a| match &a[0] {
            Value::Null => Err(Throw::no_message("java.lang.NullPointerException")),
            v => Ok(wrap(Some(v.clone()))),
        },
        ("ofNullable", 1) => |_vm, a| Ok(nullable(a[0].clone())),
        _ => return None,
    })
}

impl JObject for Optional {
    fn class_name(&self) -> &str {
        CLASS
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn java_equals(&self, other: &dyn JObject) -> bool {
        match (other.as_any().downcast_ref::<Optional>(), &self.0) {
            (Some(Optional(None)), None) => true,
            (Some(Optional(Some(b))), Some(a)) => a.java_equals(b),
            _ => false,
        }
    }

    fn java_hash(&self) -> i32 {
        self.0.as_ref().map_or(0, Value::java_hash)
    }

    fn java_to_string(&self) -> String {
        match &self.0 {
            None => "Optional.empty".to_string(),
            Some(v) => format!("Optional[{}]", v.java_to_string()),
        }
    }

    fn call(
        &self,
        vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(match (name, args) {
            ("isPresent", []) => Ok(Value::Bool(self.0.is_some())),
            ("get", []) => self
                .0
                .clone()
                .ok_or_else(|| Throw::new("java.util.NoSuchElementException", "No value present")),
            ("orElse", [other]) => Ok(self.0.clone().unwrap_or_else(|| other.clone())),
            ("orElseGet", [supplier]) => match &self.0 {
                Some(v) => Ok(v.clone()),
                None => invoke_as(vm, supplier, SUPPLIER, "get", &[]),
            },
            ("orElseThrow", [supplier]) => match &self.0 {
                Some(v) => Ok(v.clone()),
                None => invoke_as(vm, supplier, SUPPLIER, "get", &[]).and_then(|e| Err(Throw(e))),
            },
            ("ifPresent", [consumer]) => match &self.0 {
                Some(v) => invoke_as(vm, consumer, CONSUMER, "accept", std::slice::from_ref(v))
                    .map(|_| Value::Null),
                None => Ok(Value::Null),
            },
            ("map", [f]) => match &self.0 {
                Some(v) => {
                    invoke_as(vm, f, FUNCTION, "apply", std::slice::from_ref(v)).map(nullable)
                }
                None => Ok(wrap(None)),
            },
            ("flatMap", [f]) => match &self.0 {
                Some(v) => invoke_as(vm, f, FUNCTION, "apply", std::slice::from_ref(v)),
                None => Ok(wrap(None)),
            },
            ("filter", [p]) => match &self.0 {
                Some(v) => {
                    invoke_as(vm, p, PREDICATE, "test", std::slice::from_ref(v)).map(|keep| {
                        if matches!(keep, Value::Bool(true)) {
                            wrap(Some(v.clone()))
                        } else {
                            wrap(None)
                        }
                    })
                }
                None => Ok(wrap(None)),
            },
            _ => return None,
        })
    }
}
