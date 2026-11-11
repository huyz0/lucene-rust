//! Java arrays and `java.util.Arrays`.

use crate::api::{int_arg, npe};
use crate::types::{convert, prim};
use crate::value::{JArray, ListKind, Value};
use crate::vm::{ApiFn, Throw, Vm};

fn array(v: &Value) -> Result<&std::rc::Rc<std::cell::RefCell<JArray>>, Throw> {
    match v {
        Value::Array(a) => Ok(a),
        Value::Null => Err(npe()),
        o => Err(Throw::class_cast(format!(
            "{} is not an array",
            o.class_name()
        ))),
    }
}

pub fn length(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    Ok(Value::Int(array(&args[0])?.borrow().items.len() as i32))
}

pub fn load(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let a = array(&args[0])?.borrow();
    let i = int_arg(&args[1])?;
    a.items
        .get(usize::try_from(i).unwrap_or(usize::MAX))
        .cloned()
        .ok_or_else(|| {
            Throw::index(
                "java.lang.ArrayIndexOutOfBoundsException",
                i64::from(i),
                a.items.len(),
            )
        })
}

/// The element a store puts into an array of `component`.
fn element(component: &str, v: &Value) -> Result<Value, Throw> {
    match prim(component) {
        Some(p) => {
            if v.is_null() {
                return Err(npe());
            }
            Ok(convert(v, p).unwrap_or_else(|| v.clone()))
        }
        None => Ok(v.clone()),
    }
}

pub fn store(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let mut a = array(&args[0])?.borrow_mut();
    let i = int_arg(&args[1])?;
    let len = a.items.len();
    let v = element(&a.component, &args[2])?;
    let slot = a
        .items
        .get_mut(usize::try_from(i).unwrap_or(usize::MAX))
        .ok_or_else(|| {
            Throw::index(
                "java.lang.ArrayIndexOutOfBoundsException",
                i64::from(i),
                len,
            )
        })?;
    *slot = v;
    Ok(Value::Null)
}

/// A negative index counts from the end of the array.
pub fn flip_index(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let i = int_arg(&args[1])?;
    if i < 0 {
        let len = array(&args[0])?.borrow().items.len() as i32;
        return Ok(Value::Int(i.wrapping_add(len)));
    }
    Ok(Value::Int(i))
}

pub fn default_element(component: &str) -> Value {
    match prim(component) {
        Some(p) => convert(&Value::Int(0), p).unwrap_or(Value::Bool(false)),
        None => Value::Null,
    }
}

/// `new T[] {a, b, c}`.
pub fn from_items(component: &str, items: &[Value]) -> Result<Value, Throw> {
    let items = items
        .iter()
        .map(|v| element(component, v))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::array(component, items))
}

/// `new T[n][m]...`: `dims` are the sizes, outermost first; `component` is
/// the component type of the outermost array.
pub fn new_dims(component: &str, dims: &[Value]) -> Result<Value, Throw> {
    let n = int_arg(&dims[0])?;
    if n < 0 {
        return Err(Throw::new(
            "java.lang.NegativeArraySizeException",
            n.to_string(),
        ));
    }
    let n = n as usize;
    if n > 1 << 28 {
        return Err(Throw::new(
            "java.lang.OutOfMemoryError",
            "Requested array size exceeds VM limit",
        ));
    }
    let items = if dims.len() > 1 {
        let inner = component
            .strip_prefix('[')
            .map(component_name)
            .unwrap_or_default();
        (0..n)
            .map(|_| new_dims(&inner, &dims[1..]))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![default_element(component); n]
    };
    Ok(Value::array(component, items))
}

/// The binary name of the type a descriptor names (`I` → `int`).
fn component_name(desc: &str) -> String {
    match desc {
        "Z" => "boolean".into(),
        "B" => "byte".into(),
        "S" => "short".into(),
        "C" => "char".into(),
        "I" => "int".into(),
        "J" => "long".into(),
        "F" => "float".into(),
        "D" => "double".into(),
        s if s.starts_with('[') => s.to_string(),
        s => s.trim_start_matches('L').trim_end_matches(';').to_string(),
    }
}

/// The elements of an array value.
pub fn items(v: &Value) -> Result<Vec<Value>, Throw> {
    Ok(array(v)?.borrow().items.clone())
}

pub fn arrays_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("asList", 1) => |_vm, a| Ok(Value::list(ListKind::FixedSize, items(&a[0])?)),
        ("deepToString", 1) => |_vm, a| Ok(Value::string(deep_to_string(&a[0]))),
        ("deepEquals", 2) => |_vm, a| Ok(Value::Bool(deep_equals(&a[0], &a[1]))),
        ("deepHashCode", 1) => |_vm, a| Ok(Value::Int(deep_hash(&a[0]))),
        _ => return None,
    })
}

fn deep_to_string(v: &Value) -> String {
    match v {
        Value::Array(a) => {
            let a = a.borrow();
            let parts: Vec<String> = a.items.iter().map(deep_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        o => o.java_to_string(),
    }
}

fn deep_equals(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            let (x, y) = (x.borrow(), y.borrow());
            x.items.len() == y.items.len()
                && x.items.iter().zip(&y.items).all(|(p, q)| deep_equals(p, q))
        }
        _ => a.java_equals(b),
    }
}

fn deep_hash(v: &Value) -> i32 {
    match v {
        Value::Array(a) => a
            .borrow()
            .items
            .iter()
            .fold(1i32, |h, x| h.wrapping_mul(31).wrapping_add(deep_hash(x))),
        Value::Null => 0,
        o => o.java_hash(),
    }
}
