//! `List` (and the `Deque`/`Queue` methods of a `LinkedList`).

use crate::api::{self, collection, int_arg, npe};
use crate::lambda::invoke;
use crate::value::{ListKind, Value};
use crate::vm::{ApiFn, Throw, Vm};

fn oob(i: i32, len: usize) -> Throw {
    Throw::index("java.lang.IndexOutOfBoundsException", i64::from(i), len)
}

fn index(i: i32, len: usize) -> Result<usize, Throw> {
    usize::try_from(i)
        .ok()
        .filter(|&u| u < len)
        .ok_or_else(|| oob(i, len))
}

pub fn get(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let Value::List(l) = &args[0] else {
        return Err(npe());
    };
    let l = l.borrow();
    let i = int_arg(&args[1])?;
    Ok(l.items[index(i, l.items.len())?].clone())
}

/// `list.set(i, v)`: the previous element.
pub fn set(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let Value::List(l) = &args[0] else {
        return Err(npe());
    };
    if l.borrow().kind == ListKind::Immutable {
        return Err(Throw::no_message("java.lang.UnsupportedOperationException"));
    }
    let mut l = l.borrow_mut();
    let i = int_arg(&args[1])?;
    let at = index(i, l.items.len())?;
    Ok(std::mem::replace(&mut l.items[at], args[2].clone()))
}

pub fn flip_index(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let i = int_arg(&args[1])?;
    if i < 0 {
        let len = match &args[0] {
            Value::List(l) => l.borrow().items.len(),
            Value::Set(s) => s.borrow().map.len(),
            Value::Null => return Err(npe()),
            _ => 0,
        };
        return Ok(Value::Int(i.wrapping_add(len as i32)));
    }
    Ok(Value::Int(i))
}

/// `[a, b, c]`.
pub fn new_array_list_of(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    Ok(Value::list(ListKind::ArrayList, args.to_vec()))
}

pub fn constructor(class: &str, arity: usize) -> Option<ApiFn> {
    let linked = class == "java.util.LinkedList" || class == "java.util.ArrayDeque";
    Some(match (linked, arity) {
        (false, 0) => |_vm, _a| Ok(Value::list(ListKind::ArrayList, Vec::new())),
        (false, 1) => |vm, a| match &a[0] {
            Value::Int(n) if *n < 0 => Err(Throw::illegal_arg(format!("Illegal Capacity: {n}"))),
            Value::Int(_) => Ok(Value::list(ListKind::ArrayList, Vec::new())),
            c => Ok(Value::list(
                ListKind::ArrayList,
                collection::elements(vm, c)?,
            )),
        },
        (true, 0) => |_vm, _a| Ok(Value::list(ListKind::LinkedList, Vec::new())),
        (true, 1) => |vm, a| match &a[0] {
            Value::Int(_) => Ok(Value::list(ListKind::LinkedList, Vec::new())),
            c => Ok(Value::list(
                ListKind::LinkedList,
                collection::elements(vm, c)?,
            )),
        },
        _ => return None,
    })
}

pub fn static_lookup(name: &str, _arity: usize) -> Option<ApiFn> {
    Some(match name {
        "of" => |_vm, a| {
            if a.iter().any(Value::is_null) {
                return Err(npe());
            }
            Ok(Value::list(ListKind::Immutable, a.to_vec()))
        },
        "copyOf" => |vm, a| {
            Ok(Value::list(
                ListKind::Immutable,
                collection::elements(vm, &a[0])?,
            ))
        },
        _ => return None,
    })
}

fn no_such() -> Throw {
    Throw::no_message("java.util.NoSuchElementException")
}

/// Methods of `List`, then of `Collection`.
pub fn call(vm: &mut Vm, recv: &Value, name: &str, args: &[Value]) -> Option<Result<Value, Throw>> {
    let Value::List(l) = recv else { return None };
    let r = (|| -> Option<Result<Value, Throw>> {
        Some(match (name, args.len()) {
            ("get", 1) => get(vm, &[recv.clone(), args[0].clone()]),
            ("set", 2) => set(vm, &[recv.clone(), args[0].clone(), args[1].clone()]),
            ("getLength", 0) => Ok(Value::Int(l.borrow().items.len() as i32)),
            ("add", 2) => (|| {
                collection::check_mutable(recv)?;
                let i = int_arg(&args[0])?;
                let mut l = l.borrow_mut();
                let len = l.items.len();
                let at = usize::try_from(i)
                    .ok()
                    .filter(|&u| u <= len)
                    .ok_or_else(|| oob(i, len))?;
                l.items.insert(at, args[1].clone());
                Ok(Value::Null)
            })(),
            ("addAll", 2) => (|| {
                collection::check_mutable(recv)?;
                let i = int_arg(&args[0])?;
                let items = collection::elements(vm, &args[1])?;
                let mut l = l.borrow_mut();
                let len = l.items.len();
                let at = usize::try_from(i)
                    .ok()
                    .filter(|&u| u <= len)
                    .ok_or_else(|| oob(i, len))?;
                let changed = !items.is_empty();
                l.items.splice(at..at, items);
                Ok(Value::Bool(changed))
            })(),
            ("remove", 1) => (|| {
                collection::check_mutable(recv)?;
                match &args[0] {
                    Value::Int(_) | Value::Short(_) | Value::Byte(_) | Value::Char(_) => {
                        let i = int_arg(&args[0])?;
                        let mut l = l.borrow_mut();
                        let at = index(i, l.items.len())?;
                        Ok(l.items.remove(at))
                    }
                    other => collection::remove_object(vm, recv, other),
                }
            })(),
            ("indexOf", 1) => Ok(Value::Int(
                l.borrow()
                    .items
                    .iter()
                    .position(|x| x.java_equals(&args[0]))
                    .map_or(-1, |i| i as i32),
            )),
            ("lastIndexOf", 1) => Ok(Value::Int(
                l.borrow()
                    .items
                    .iter()
                    .rposition(|x| x.java_equals(&args[0]))
                    .map_or(-1, |i| i as i32),
            )),
            ("sort", 1) => (|| {
                let mut items = l.borrow().items.clone();
                collection::sort_values(vm, &mut items, Some(&args[0]))?;
                if l.borrow().kind == ListKind::Immutable && items.len() > 1 {
                    return Err(Throw::no_message("java.lang.UnsupportedOperationException"));
                }
                l.borrow_mut().items = items;
                Ok(Value::Null)
            })(),
            ("replaceAll", 1) => (|| {
                let items = l.borrow().items.clone();
                let mut out = Vec::with_capacity(items.len());
                for x in items {
                    out.push(invoke(vm, &args[0], &[x])?);
                }
                l.borrow_mut().items = out;
                Ok(Value::Null)
            })(),
            ("subList", 2) => (|| {
                let (a, b) = (int_arg(&args[0])?, int_arg(&args[1])?);
                let l = l.borrow();
                let n = l.items.len() as i32;
                if a < 0 || b > n || a > b {
                    return Err(Throw::new(
                        "java.lang.IndexOutOfBoundsException",
                        format!("fromIndex: {a}, toIndex: {b}, size: {n}"),
                    ));
                }
                Ok(Value::list(
                    ListKind::ArrayList,
                    l.items[a as usize..b as usize].to_vec(),
                ))
            })(),
            ("equals", 1) => Ok(Value::Bool(recv.java_equals(&args[0]))),
            ("hashCode", 0) => Ok(Value::Int(recv.java_hash())),
            ("clone", 0) => Ok(Value::list(l.borrow().kind, l.borrow().items.clone())),
            ("trimToSize", 0) | ("ensureCapacity", 1) => Ok(Value::Null),
            ("getByPath", 1) | ("getByPath", 2) => api::map::get_by_path(vm, recv, args),
            // Deque and Queue, for a LinkedList.
            ("addFirst", 1) | ("offerFirst", 1) | ("push", 1) => (|| {
                collection::check_mutable(recv)?;
                l.borrow_mut().items.insert(0, args[0].clone());
                Ok(if name == "offerFirst" {
                    Value::Bool(true)
                } else {
                    Value::Null
                })
            })(),
            ("addLast", 1) | ("offerLast", 1) | ("offer", 1) => (|| {
                collection::check_mutable(recv)?;
                l.borrow_mut().items.push(args[0].clone());
                Ok(if name == "addLast" {
                    Value::Null
                } else {
                    Value::Bool(true)
                })
            })(),
            ("getFirst", 0) | ("element", 0) => {
                l.borrow().items.first().cloned().ok_or_else(no_such)
            }
            ("getLast", 0) => l.borrow().items.last().cloned().ok_or_else(no_such),
            ("peek", 0) | ("peekFirst", 0) => {
                Ok(l.borrow().items.first().cloned().unwrap_or_default())
            }
            ("peekLast", 0) => Ok(l.borrow().items.last().cloned().unwrap_or_default()),
            ("poll", 0) | ("pollFirst", 0) => {
                let mut l = l.borrow_mut();
                Ok(if l.items.is_empty() {
                    Value::Null
                } else {
                    l.items.remove(0)
                })
            }
            ("pollLast", 0) => Ok(l.borrow_mut().items.pop().unwrap_or_default()),
            ("pop", 0) | ("removeFirst", 0) | ("remove", 0) => {
                let mut l = l.borrow_mut();
                if l.items.is_empty() {
                    Err(no_such())
                } else {
                    Ok(l.items.remove(0))
                }
            }
            ("removeLast", 0) => l.borrow_mut().items.pop().ok_or_else(no_such),
            _ => return None,
        })
    })();
    r.or_else(|| collection::call(vm, recv, name, args))
}
