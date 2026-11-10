//! `Object`'s methods for every receiver, `java.util.Objects`,
//! `System`, and exceptions a script constructs.

use crate::api::{int_arg, npe, str_arg};
use crate::compare;
use crate::lambda::invoke;
use crate::types::exception_parent;
use crate::value::Value;
use crate::vm::{ApiFn, JException, Throw, Vm};

/// `equals`, `hashCode`, `toString` and `compareTo` on any receiver.
pub fn object_call(
    _vm: &mut Vm,
    recv: &Value,
    name: &str,
    args: &[Value],
) -> Option<Result<Value, Throw>> {
    Some(match (name, args.len()) {
        ("equals", 1) => Ok(Value::Bool(recv.java_equals(&args[0]))),
        ("hashCode", 0) => Ok(Value::Int(recv.java_hash())),
        ("toString", 0) => Ok(Value::string(recv.java_to_string())),
        ("compareTo", 1) if crate::types::instance_of(recv, "java.lang.Comparable") => {
            compare::compare_to(recv, &args[0]).map(|o| {
                Value::Int(match (recv, &args[0]) {
                    (Value::Str(a), Value::Str(b)) => crate::api::string::java_compare(a, b),
                    _ => compare::ordering_int(o),
                })
            })
        }
        _ => return None,
    })
}

pub fn objects_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("equals", 2) => |_vm, a| Ok(Value::Bool(a[0].java_equals(&a[1]))),
        ("deepEquals", 2) => |_vm, a| Ok(Value::Bool(a[0].java_equals(&a[1]))),
        ("hashCode", 1) => |_vm, a| Ok(Value::Int(a[0].java_hash())),
        ("hash", 1) => |_vm, a| {
            let items = match &a[0] {
                Value::Array(arr) => arr.borrow().items.clone(),
                v => vec![v.clone()],
            };
            Ok(Value::Int(items.iter().fold(1i32, |h, v| {
                h.wrapping_mul(31).wrapping_add(v.java_hash())
            })))
        },
        ("isNull", 1) => |_vm, a| Ok(Value::Bool(a[0].is_null())),
        ("nonNull", 1) => |_vm, a| Ok(Value::Bool(!a[0].is_null())),
        ("toString", 1) => |_vm, a| Ok(Value::string(a[0].java_to_string())),
        ("toString", 2) => |_vm, a| {
            Ok(if a[0].is_null() {
                a[1].clone()
            } else {
                Value::string(a[0].java_to_string())
            })
        },
        ("requireNonNull", 1) => |_vm, a| {
            if a[0].is_null() {
                Err(npe())
            } else {
                Ok(a[0].clone())
            }
        },
        ("requireNonNull", 2) => |_vm, a| {
            if a[0].is_null() {
                Err(Throw::npe(
                    str_arg(&a[1]).map(|s| s.to_string()).unwrap_or_default(),
                ))
            } else {
                Ok(a[0].clone())
            }
        },
        ("compare", 3) => |vm, a| {
            if crate::def::identical(&a[0], &a[1]) {
                return Ok(Value::Int(0));
            }
            invoke(vm, &a[2], &[a[0].clone(), a[1].clone()])
                .and_then(|v| int_arg(&v).map(Value::Int))
        },
        _ => return None,
    })
}

pub fn system_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("currentTimeMillis", 0) => |_vm, _a| {
            let ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            Ok(Value::Long(ms))
        },
        ("nanoTime", 0) => |_vm, _a| {
            let ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            Ok(Value::Long(ns))
        },
        ("identityHashCode", 1) => |_vm, a| Ok(Value::Int(a[0].java_hash())),
        _ => return None,
    })
}

/// Builds an exception object of `class` from constructor arguments.
pub fn exception_of(class: &str, args: &[Value]) -> Result<Value, Throw> {
    if exception_parent(class).is_none() && class != "java.lang.Exception" {
        return Err(Throw::new(
            crate::api::UNSUPPORTED,
            format!("constructor {class}"),
        ));
    }
    let message = match args.first() {
        None => None,
        Some(Value::Null) => None,
        Some(Value::Str(s)) => Some(s.to_string()),
        Some(v) => Some(v.java_to_string()),
    };
    Ok(Value::object(JException {
        class: class.to_string(),
        message,
        cause: None,
    }))
}
