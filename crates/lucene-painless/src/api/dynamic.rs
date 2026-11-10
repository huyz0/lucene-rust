//! Calls, field reads and index reads on a receiver whose class is known
//! only at run time (`def`), as Painless's `Def` resolves them.

use crate::api::{self, npe};
use crate::types::canonical;
use crate::value::Value;
use crate::vm::{Throw, Vm};

fn receiver_name(v: &Value) -> String {
    canonical(&v.class_name())
}

/// `receiver.name(args)`.
pub fn call(vm: &mut Vm, recv: &Value, name: &str, args: &[Value]) -> Result<Value, Throw> {
    let r = match recv {
        Value::Null => {
            return Err(Throw::npe(format!(
                "Cannot invoke \"{name}()\" because value is null"
            )))
        }
        Value::Str(s) => api::string::call(vm, s, name, args),
        Value::List(_) => api::list::call(vm, recv, name, args),
        Value::Map(_) => api::map::call(vm, recv, name, args),
        Value::Set(_) => api::collection::set_call(vm, recv, name, args),
        Value::Array(_) => None,
        Value::Object(o) => o.clone().call(vm, recv, name, args),
        _ => api::number::call(vm, recv, name, args),
    };
    if let Some(r) = r {
        return r;
    }
    if let Some(r) = api::objects::object_call(vm, recv, name, args) {
        return r;
    }
    Err(Throw::new(
        api::UNSUPPORTED,
        format!(
            "dynamic method [{}, {name}/{}] not found",
            receiver_name(recv),
            args.len()
        ),
    ))
}

/// A negative index into an array or list counts from its end; a map key
/// is left alone.
pub fn flip_index(vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let len = match &args[0] {
        Value::Array(a) => a.borrow().items.len(),
        Value::List(l) => l.borrow().items.len(),
        Value::Object(o) if o.instance_of("java.util.List") => {
            api::int_arg(&call(vm, &args[0], "size", &[])?)?.max(0) as usize
        }
        _ => return Ok(args[1].clone()),
    };
    match &args[1] {
        Value::Int(i) if *i < 0 => Ok(Value::Int(i.wrapping_add(len as i32))),
        Value::Long(i) if *i < 0 => Ok(Value::Int((*i as i32).wrapping_add(len as i32))),
        v => Ok(v.clone()),
    }
}

/// `receiver[index]` on a `def` receiver.
pub fn brace_load(vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    match &args[0] {
        Value::Array(_) => api::array::load(vm, args),
        Value::List(_) => api::list::get(vm, args),
        Value::Map(m) => Ok(api::map::get_value(&m.borrow(), &args[1])),
        Value::Null => Err(npe()),
        Value::Object(o) if o.instance_of("java.util.List") || o.instance_of("java.util.Map") => {
            call(vm, &args[0], "get", &args[1..2])
        }
        other => Err(Throw::illegal_arg(format!(
            "Attempting to address a non-array-like type [{}] as an array.",
            receiver_name(other)
        ))),
    }
}

/// `receiver[index] = value` on a `def` receiver.
pub fn brace_store(vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    match &args[0] {
        Value::Array(_) => api::array::store(vm, args),
        Value::List(_) => api::list::set(vm, args).map(|_| Value::Null),
        Value::Map(_) => api::map::put(vm, &args[0], &args[1], &args[2]).map(|_| Value::Null),
        Value::Null => Err(npe()),
        Value::Object(o) if o.instance_of("java.util.List") => {
            call(vm, &args[0], "set", &args[1..3]).map(|_| Value::Null)
        }
        Value::Object(o) if o.instance_of("java.util.Map") => {
            call(vm, &args[0], "put", &args[1..3]).map(|_| Value::Null)
        }
        other => Err(Throw::illegal_arg(format!(
            "Attempting to address a non-array-like type [{}] as an array.",
            receiver_name(other)
        ))),
    }
}

fn getter_names(name: &str) -> [String; 2] {
    // The getters scripts read most, without formatting their names.
    match name {
        "value" => return ["getValue".into(), "isValue".into()],
        "values" => return ["getValues".into(), "isValues".into()],
        _ => {}
    }
    let mut c = name.chars();
    let cap = match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    };
    [format!("get{cap}"), format!("is{cap}")]
}

/// `receiver.name` on a `def` receiver.
pub fn load_dot(vm: &mut Vm, recv: &Value, name: &str) -> Result<Value, Throw> {
    match recv {
        Value::Null => Err(npe()),
        Value::Map(m) if name == "empty" => Ok(Value::Bool(m.borrow().map.is_empty())),
        Value::List(l) if name == "empty" => Ok(Value::Bool(l.borrow().items.is_empty())),
        Value::List(l) if name == "length" => Ok(Value::Int(l.borrow().items.len() as i32)),
        Value::Map(m) => Ok(api::map::get_value(&m.borrow(), &Value::str(name))),
        Value::Array(a) if name == "length" => Ok(Value::Int(a.borrow().items.len() as i32)),
        Value::List(_) => match name.parse::<i32>() {
            Ok(i) => api::list::get(vm, &[recv.clone(), Value::Int(i)]),
            Err(_) => getter(vm, recv, name),
        },
        Value::Object(o) => match o.clone().field(vm, name) {
            Some(r) => r,
            None => getter(vm, recv, name),
        },
        _ => getter(vm, recv, name),
    }
}

/// A public field of a host object (`ZonedDateTime` has none; doc values
/// views expose some).
pub fn load_field(vm: &mut Vm, recv: &Value, name: &str) -> Result<Value, Throw> {
    load_dot(vm, recv, name)
}

fn getter(vm: &mut Vm, recv: &Value, name: &str) -> Result<Value, Throw> {
    for g in getter_names(name) {
        match call(vm, recv, &g, &[]) {
            Err(t) if t.class() == api::UNSUPPORTED => continue,
            r => return r,
        }
    }
    Err(Throw::illegal_arg(format!(
        "dynamic getter [{}, {name}] not found",
        receiver_name(recv)
    )))
}

/// `receiver.name = value` on a `def` receiver.
pub fn store_dot(vm: &mut Vm, recv: &Value, name: &str, value: &Value) -> Result<Value, Throw> {
    match recv {
        Value::Null => Err(npe()),
        Value::Map(_) => api::map::put(vm, recv, &Value::str(name), value).map(|_| Value::Null),
        Value::List(_) => match name.parse::<i32>() {
            Ok(i) => api::list::set(vm, &[recv.clone(), Value::Int(i), value.clone()])
                .map(|_| Value::Null),
            Err(_) => setter(vm, recv, name, value),
        },
        _ => setter(vm, recv, name, value),
    }
}

fn setter(vm: &mut Vm, recv: &Value, name: &str, value: &Value) -> Result<Value, Throw> {
    let mut c = name.chars();
    let cap = match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    };
    match call(vm, recv, &format!("set{cap}"), std::slice::from_ref(value)) {
        Err(t) if t.class() == api::UNSUPPORTED => Err(Throw::illegal_arg(format!(
            "dynamic setter [{}, {name}] not found",
            receiver_name(recv)
        ))),
        r => r.map(|_| Value::Null),
    }
}
