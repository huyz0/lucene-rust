//! Calls, field reads and index reads on a receiver whose class is known
//! only at run time (`def`), as Painless's `Def` resolves them.

use crate::api;
use crate::value::Value;
use crate::vm::{Throw, Vm};

/// `typeToCanonicalTypeName` of the receiver's class.
fn receiver_name(v: &Value) -> String {
    crate::types::java_canonical(&v.class_name())
}

/// A `def` call site's null receiver: `DefBootstrap`'s
/// `callArgs[0].getClass()`, in the JVM's words.
pub fn null_receiver() -> Throw {
    Throw::npe("Cannot invoke \"Object.getClass()\" because \"callArgs[0]\" is null")
}

/// `receiver.name(args)`: Painless's `IllegalArgumentException` when the
/// receiver's class has no such method.
pub fn call(vm: &mut Vm, recv: &Value, name: &str, args: &[Value]) -> Result<Value, Throw> {
    match recv {
        Value::Null => Err(null_receiver()),
        _ => try_call(vm, recv, name, args).unwrap_or_else(|| {
            Err(Throw::illegal_arg(format!(
                "dynamic method [{}, {name}/{}] not found",
                receiver_name(recv),
                args.len()
            )))
        }),
    }
}

/// `receiver.name(args)` when the receiver's class has the method natively;
/// `None` when it has not. A method the runtime names but lacks answers
/// [`api::UNSUPPORTED`].
pub fn try_call(
    vm: &mut Vm,
    recv: &Value,
    name: &str,
    args: &[Value],
) -> Option<Result<Value, Throw>> {
    let r = match recv {
        Value::Null => return Some(call(vm, recv, name, args)),
        Value::Str(s) => api::string::call(vm, s, name, args),
        Value::List(_) => api::list::call(vm, recv, name, args),
        Value::Map(_) => api::map::call(vm, recv, name, args),
        Value::Set(_) => api::collection::set_call(vm, recv, name, args),
        Value::Array(_) => None,
        Value::Object(o) => o.clone().call(vm, recv, name, args),
        _ => api::number::call(vm, recv, name, args),
    };
    r.or_else(|| api::objects::object_call(vm, recv, name, args))
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
        Value::Null => return Err(null_receiver()),
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
        Value::Null => Err(null_receiver()),
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
        Value::Null => Err(null_receiver()),
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

/// `receiver.name` on a `def` receiver, as `Def.lookupGetter` resolves it:
/// the class's getter (or public field), then an array's `length`, a map's
/// key, a list's index.
pub fn load_dot(vm: &mut Vm, recv: &Value, name: &str) -> Result<Value, Throw> {
    if recv.is_null() {
        return Err(null_receiver());
    }
    if let Some(r) = try_getter(vm, recv, name) {
        return r;
    }
    match recv {
        Value::Array(a) if name == "length" => Ok(Value::Int(a.borrow().items.len() as i32)),
        Value::Map(m) => Ok(api::map::get_value(&m.borrow(), &Value::str(name))),
        Value::Object(o) if o.instance_of("java.util.Map") => {
            call(vm, recv, "get", &[Value::str(name)])
        }
        Value::List(_) => list_index(name).and_then(|i| api::list::get(vm, &[recv.clone(), i])),
        Value::Object(o) if o.instance_of("java.util.List") => {
            list_index(name).and_then(|i| call(vm, recv, "get", &[i]))
        }
        _ => Err(Throw::illegal_arg(format!(
            "dynamic getter [{}, {name}] not found",
            receiver_name(recv)
        ))),
    }
}

fn list_index(name: &str) -> Result<Value, Throw> {
    name.parse::<i32>()
        .map(Value::Int)
        .map_err(|_| Throw::illegal_arg(format!("Illegal list shortcut value [{name}].")))
}

/// The receiver class's getter `name` (`getName()`, `isName()`) or public
/// field, when it has one natively.
pub fn try_getter(vm: &mut Vm, recv: &Value, name: &str) -> Option<Result<Value, Throw>> {
    if let Value::Object(o) = recv {
        if let Some(r) = o.clone().field(vm, name) {
            return Some(r);
        }
    }
    let [get, is] = getter_names(name);
    try_call(vm, recv, &get, &[]).or_else(|| try_call(vm, recv, &is, &[]))
}

/// A public field of a host object (`ZonedDateTime` has none; doc values
/// views expose some).
pub fn load_field(vm: &mut Vm, recv: &Value, name: &str) -> Result<Value, Throw> {
    load_dot(vm, recv, name)
}

/// `receiver.name = value` on a `def` receiver, as `Def.lookupSetter`
/// resolves it: the class's setter (or public field), then a map's key, a
/// list's index.
pub fn store_dot(vm: &mut Vm, recv: &Value, name: &str, value: &Value) -> Result<Value, Throw> {
    if recv.is_null() {
        return Err(null_receiver());
    }
    if let Some(r) = try_setter(vm, recv, name, value) {
        return r.map(|_| Value::Null);
    }
    match recv {
        Value::Map(_) => api::map::put(vm, recv, &Value::str(name), value).map(|_| Value::Null),
        Value::Object(o) if o.instance_of("java.util.Map") => {
            call(vm, recv, "put", &[Value::str(name), value.clone()]).map(|_| Value::Null)
        }
        Value::List(_) => list_index(name).and_then(|i| {
            api::list::set(vm, &[recv.clone(), i, value.clone()]).map(|_| Value::Null)
        }),
        Value::Object(o) if o.instance_of("java.util.List") => list_index(name)
            .and_then(|i| call(vm, recv, "set", &[i, value.clone()]).map(|_| Value::Null)),
        _ => Err(Throw::illegal_arg(format!(
            "dynamic setter [{}, {name}] not found",
            receiver_name(recv)
        ))),
    }
}

/// The receiver class's setter `name` (`setName(value)`), when it has one
/// natively.
pub fn try_setter(
    vm: &mut Vm,
    recv: &Value,
    name: &str,
    value: &Value,
) -> Option<Result<Value, Throw>> {
    try_call(
        vm,
        recv,
        &format!("set{}", capitalized(name)),
        std::slice::from_ref(value),
    )
}

fn capitalized(name: &str) -> String {
    let mut c = name.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}
