//! `Iterable` and `Collection` methods shared by lists and sets (with
//! Painless's Groovy-style augmentations), `Set`, and `Collections`.

use std::cmp::Ordering;

use crate::api::{self, int_arg, npe, str_arg};
use crate::compare;
use crate::jmap::{JMap, Order};
use crate::lambda::invoke;
use crate::value::{ListKind, MapKind, SetKind, Value};
use crate::vm::{ApiFn, Throw, Vm};

fn unsupported_op() -> Throw {
    Throw::no_message("java.lang.UnsupportedOperationException")
}

/// The elements of a list or set (or any iterable) in iteration order.
pub fn elements(vm: &mut Vm, v: &Value) -> Result<Vec<Value>, Throw> {
    api::iter::elements(vm, v)
}

/// Whether a list or set can change.
pub fn check_mutable(v: &Value) -> Result<(), Throw> {
    let ok = match v {
        Value::List(l) => matches!(l.borrow().kind, ListKind::ArrayList | ListKind::LinkedList),
        Value::Set(s) => s.borrow().kind != SetKind::Immutable,
        Value::Map(m) => m.borrow().kind != MapKind::Immutable,
        _ => true,
    };
    if ok {
        Ok(())
    } else {
        Err(unsupported_op())
    }
}

/// `collection.add(x)`.
pub fn add(v: &Value, x: Value) -> Result<bool, Throw> {
    check_mutable(v)?;
    match v {
        Value::List(l) => {
            l.borrow_mut().items.push(x);
            Ok(true)
        }
        Value::Set(s) => {
            let mut s = s.borrow_mut();
            if s.kind == SetKind::TreeSet {
                tree_key_check(&s.map, &x)?;
            }
            Ok(s.map.insert(x, Value::Null).is_none())
        }
        Value::Null => Err(npe()),
        o => Err(Throw::class_cast(format!(
            "{} is not a collection",
            o.class_name()
        ))),
    }
}

/// A `TreeSet`/`TreeMap` key must be comparable with the existing keys.
pub fn tree_key_check(map: &JMap, key: &Value) -> Result<(), Throw> {
    if key.is_null() {
        return Err(npe());
    }
    match map.iter().next() {
        Some((k, _)) => compare::compare_to(key, k).map(|_| ()),
        None => compare::compare_to(key, key).map(|_| ()),
    }
}

/// `collection.contains(x)`.
pub fn contains(v: &Value, x: &Value) -> Result<bool, Throw> {
    Ok(match v {
        Value::List(l) => l.borrow().items.iter().any(|e| e.java_equals(x)),
        Value::Set(s) => s.borrow().map.contains_key(x),
        Value::Null => return Err(npe()),
        o => {
            return Err(Throw::class_cast(format!(
                "{} is not a collection",
                o.class_name()
            )))
        }
    })
}

/// `Collection.remove(Object)`.
pub fn remove_object(_vm: &mut Vm, v: &Value, x: &Value) -> Result<Value, Throw> {
    check_mutable(v)?;
    Ok(Value::Bool(match v {
        Value::List(l) => {
            let mut l = l.borrow_mut();
            match l.items.iter().position(|e| e.java_equals(x)) {
                Some(i) => {
                    l.items.remove(i);
                    true
                }
                None => false,
            }
        }
        Value::Set(s) => s.borrow_mut().map.remove(x).is_some(),
        Value::Map(m) => return Ok(m.borrow_mut().map.remove(x).unwrap_or_default()),
        Value::Null => return Err(npe()),
        o => {
            return Err(Throw::class_cast(format!(
                "{} is not a collection",
                o.class_name()
            )))
        }
    }))
}

fn element_count(v: &Value) -> usize {
    match v {
        Value::List(l) => l.borrow().items.len(),
        Value::Set(s) => s.borrow().map.len(),
        _ => 0,
    }
}

fn truthy(v: &Value) -> Result<bool, Throw> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Null => Err(npe()),
        o => Err(Throw::cast_failure(&o.class_name(), "java.lang.Boolean")),
    }
}

/// Replaces the contents of a list or set with `items`.
fn replace_contents(v: &Value, items: Vec<Value>) {
    match v {
        Value::List(l) => l.borrow_mut().items = items,
        Value::Set(s) => {
            let mut s = s.borrow_mut();
            s.map.clear();
            for x in items {
                s.map.insert(x, Value::Null);
            }
        }
        _ => {}
    }
}

pub fn new_list(items: Vec<Value>) -> Value {
    Value::list(ListKind::ArrayList, items)
}

/// The methods of `Iterable` and `Collection`, for a list or set receiver.
pub fn call(vm: &mut Vm, recv: &Value, name: &str, args: &[Value]) -> Option<Result<Value, Throw>> {
    Some((|| -> Result<Value, Throw> {
        Ok(match (name, args.len()) {
            ("size", 0) => Value::Int(element_count(recv) as i32),
            ("isEmpty", 0) => Value::Bool(element_count(recv) == 0),
            ("add", 1) => Value::Bool(add(recv, args[0].clone())?),
            ("contains", 1) => Value::Bool(contains(recv, &args[0])?),
            ("clear", 0) => {
                check_mutable(recv)?;
                replace_contents(recv, Vec::new());
                Value::Null
            }
            ("addAll", 1) => {
                let items = elements(vm, &args[0])?;
                let mut changed = false;
                for x in items {
                    changed |= add(recv, x)?;
                }
                Value::Bool(changed)
            }
            ("containsAll", 1) => {
                let items = elements(vm, &args[0])?;
                let mut all = true;
                for x in &items {
                    all &= contains(recv, x)?;
                }
                Value::Bool(all)
            }
            ("removeAll", 1) | ("retainAll", 1) => {
                check_mutable(recv)?;
                let keep_matching = name == "retainAll";
                let before = elements(vm, recv)?;
                let mut kept = Vec::with_capacity(before.len());
                for x in &before {
                    if contains(&args[0], x)? == keep_matching {
                        kept.push(x.clone());
                    }
                }
                let changed = kept.len() != before.len();
                replace_contents(recv, kept);
                Value::Bool(changed)
            }
            ("removeIf", 1) => {
                check_mutable(recv)?;
                let before = elements(vm, recv)?;
                let mut kept = Vec::with_capacity(before.len());
                for x in &before {
                    if !truthy(&invoke(vm, &args[0], std::slice::from_ref(x))?)? {
                        kept.push(x.clone());
                    }
                }
                let changed = kept.len() != before.len();
                replace_contents(recv, kept);
                Value::Bool(changed)
            }
            ("toArray", 0) => Value::array("java.lang.Object", elements(vm, recv)?),
            ("toArray", 1) => {
                let comp = match &args[0] {
                    Value::Array(a) => a.borrow().component.to_string(),
                    _ => "java.lang.Object".to_string(),
                };
                Value::array(&comp, elements(vm, recv)?)
            }
            ("iterator", 0) => api::iter::JIter::value(elements(vm, recv)?),
            ("forEach", 1) | ("each", 1) => {
                for x in elements(vm, recv)? {
                    invoke(vm, &args[0], &[x])?;
                }
                if name == "each" {
                    recv.clone()
                } else {
                    Value::Null
                }
            }
            ("eachWithIndex", 1) => {
                for (i, x) in elements(vm, recv)?.into_iter().enumerate() {
                    invoke(vm, &args[0], &[x, Value::Int(i as i32)])?;
                }
                recv.clone()
            }
            ("any", 1) => {
                for x in elements(vm, recv)? {
                    if truthy(&invoke(vm, &args[0], &[x])?)? {
                        return Ok(Value::Bool(true));
                    }
                }
                Value::Bool(false)
            }
            ("every", 1) => {
                for x in elements(vm, recv)? {
                    if !truthy(&invoke(vm, &args[0], &[x])?)? {
                        return Ok(Value::Bool(false));
                    }
                }
                Value::Bool(true)
            }
            ("count", 1) => {
                let mut n = 0;
                for x in elements(vm, recv)? {
                    if truthy(&invoke(vm, &args[0], &[x])?)? {
                        n += 1;
                    }
                }
                Value::Int(n)
            }
            ("asCollection", 0) => recv.clone(),
            ("asList", 0) => match recv {
                Value::List(_) => recv.clone(),
                _ => new_list(elements(vm, recv)?),
            },
            ("findResults", 1) => {
                let mut out = Vec::new();
                for x in elements(vm, recv)? {
                    let r = invoke(vm, &args[0], &[x])?;
                    if !r.is_null() {
                        out.push(r);
                    }
                }
                new_list(out)
            }
            ("groupBy", 1) => {
                let mut m = JMap::new(Order::Hash);
                for x in elements(vm, recv)? {
                    let k = invoke(vm, &args[0], std::slice::from_ref(&x))?;
                    match m.get_mut(&k) {
                        Some(Value::List(l)) => l.borrow_mut().items.push(x),
                        _ => {
                            m.insert(k, new_list(vec![x]));
                        }
                    }
                }
                Value::map(MapKind::HashMap, m)
            }
            ("join", 1) => {
                let sep = str_arg(&args[0])?;
                let mut s = String::new();
                for (i, x) in elements(vm, recv)?.iter().enumerate() {
                    if i > 0 {
                        s.push_str(&sep);
                    }
                    x.write_java(&mut s, 0);
                }
                Value::string(s)
            }
            ("sum", 0) => {
                let mut t = 0.0;
                for x in elements(vm, recv)? {
                    t += api::double_arg(&x)?;
                }
                Value::Double(t)
            }
            ("sum", 1) => {
                let mut t = 0.0;
                for x in elements(vm, recv)? {
                    t += api::double_arg(&invoke(vm, &args[0], &[x])?)?;
                }
                Value::Double(t)
            }
            ("collect", 1) => {
                let mut out = Vec::new();
                for x in elements(vm, recv)? {
                    out.push(invoke(vm, &args[0], &[x])?);
                }
                new_list(out)
            }
            ("collect", 2) => {
                for x in elements(vm, recv)? {
                    let r = invoke(vm, &args[1], &[x])?;
                    add(&args[0], r)?;
                }
                args[0].clone()
            }
            ("find", 1) => {
                for x in elements(vm, recv)? {
                    if truthy(&invoke(vm, &args[0], std::slice::from_ref(&x))?)? {
                        return Ok(x);
                    }
                }
                Value::Null
            }
            ("findAll", 1) => {
                let mut out = Vec::new();
                for x in elements(vm, recv)? {
                    if truthy(&invoke(vm, &args[0], std::slice::from_ref(&x))?)? {
                        out.push(x);
                    }
                }
                new_list(out)
            }
            ("findResult", 1) | ("findResult", 2) => {
                let (default, f) = if args.len() == 2 { (args[0].clone(), &args[1]) } else { (Value::Null, &args[0]) };
                for x in elements(vm, recv)? {
                    let r = invoke(vm, f, &[x])?;
                    if !r.is_null() {
                        return Ok(r);
                    }
                }
                default
            }
            ("split", 1) => {
                let (mut yes, mut no) = (Vec::new(), Vec::new());
                for x in elements(vm, recv)? {
                    if truthy(&invoke(vm, &args[0], std::slice::from_ref(&x))?)? {
                        yes.push(x);
                    } else {
                        no.push(x);
                    }
                }
                new_list(vec![new_list(yes), new_list(no)])
            }
            ("stream", 0) => return Err(crate::compile::missing("Collection.stream")),
            _ => return Err(Throw::new(api::UNSUPPORTED, "")),
        })
    })())
    .filter(|r| !matches!(r, Err(t) if t.class() == api::UNSUPPORTED && t.message().as_deref() == Some("")))
}

/// Methods of `Set` (and `SortedSet`/`NavigableSet` for a `TreeSet`).
pub fn set_call(
    vm: &mut Vm,
    recv: &Value,
    name: &str,
    args: &[Value],
) -> Option<Result<Value, Throw>> {
    let Value::Set(s) = recv else { return None };
    let r = match (name, args.len()) {
        ("remove", 1) => remove_object(vm, recv, &args[0]),
        ("first", 0) | ("last", 0) => {
            let keys = s.borrow().map.keys();
            let k = if name == "first" {
                keys.first()
            } else {
                keys.last()
            };
            k.cloned()
                .ok_or_else(|| Throw::no_message("java.util.NoSuchElementException"))
        }
        ("pollFirst", 0) | ("pollLast", 0) => {
            let keys = s.borrow().map.keys();
            let k = if name == "pollFirst" {
                keys.first()
            } else {
                keys.last()
            };
            match k {
                Some(k) => {
                    s.borrow_mut().map.remove(k);
                    Ok(k.clone())
                }
                None => Ok(Value::Null),
            }
        }
        ("ceiling", 1) | ("floor", 1) | ("higher", 1) | ("lower", 1) => {
            let keys = s.borrow().map.keys();
            Ok(navigate(&keys, &args[0], name).cloned().unwrap_or_default())
        }
        ("equals", 1) => Ok(Value::Bool(recv.java_equals(&args[0]))),
        ("hashCode", 0) => Ok(Value::Int(recv.java_hash())),
        _ => return call(vm, recv, name, args),
    };
    Some(r)
}

/// `ceiling`/`floor`/`higher`/`lower` over sorted `keys`.
pub fn navigate<'a>(keys: &'a [Value], k: &Value, how: &str) -> Option<&'a Value> {
    let cmp = |x: &Value| compare::natural(x, k).unwrap_or(Ordering::Equal);
    match how {
        "ceiling" => keys.iter().find(|x| cmp(x) != Ordering::Less),
        "higher" => keys.iter().find(|x| cmp(x) == Ordering::Greater),
        "floor" => keys.iter().rev().find(|x| cmp(x) != Ordering::Greater),
        _ => keys.iter().rev().find(|x| cmp(x) == Ordering::Less),
    }
}

pub fn new_set(kind: SetKind, items: Vec<Value>) -> Result<Value, Throw> {
    let order = match kind {
        SetKind::HashSet | SetKind::Immutable => Order::Hash,
        SetKind::LinkedHashSet => Order::Insertion,
        SetKind::TreeSet => Order::Sorted,
    };
    let mut m = JMap::new(order);
    for x in items {
        if kind == SetKind::TreeSet {
            tree_key_check(&m, &x)?;
        }
        m.insert(x, Value::Null);
    }
    Ok(Value::set(kind, m))
}

pub fn set_constructor(class: &str, arity: usize) -> Option<ApiFn> {
    Some(match (class, arity) {
        ("java.util.HashSet", 0) => |_vm, _a| new_set(SetKind::HashSet, Vec::new()),
        ("java.util.HashSet", 1) => |vm, a| match &a[0] {
            Value::Int(_) => new_set(SetKind::HashSet, Vec::new()),
            c => {
                let items = elements(vm, c)?;
                let mut m = JMap::for_copy(Order::Hash, items.len());
                for x in items {
                    m.insert(x, Value::Null);
                }
                Ok(Value::set(SetKind::HashSet, m))
            }
        },
        ("java.util.LinkedHashSet", 0) => |_vm, _a| new_set(SetKind::LinkedHashSet, Vec::new()),
        ("java.util.LinkedHashSet", 1) => |vm, a| match &a[0] {
            Value::Int(_) => new_set(SetKind::LinkedHashSet, Vec::new()),
            c => new_set(SetKind::LinkedHashSet, elements(vm, c)?),
        },
        ("java.util.TreeSet", 0) => |_vm, _a| new_set(SetKind::TreeSet, Vec::new()),
        ("java.util.TreeSet", 1) => |vm, a| new_set(SetKind::TreeSet, elements(vm, &a[0])?),
        _ => return None,
    })
}

pub fn set_static_lookup(name: &str, _arity: usize) -> Option<ApiFn> {
    Some(match name {
        "of" => |_vm, a| new_set(SetKind::Immutable, a.to_vec()),
        "copyOf" => |vm, a| new_set(SetKind::Immutable, elements(vm, &a[0])?),
        _ => return None,
    })
}

fn list_ref(v: &Value) -> Result<&std::rc::Rc<std::cell::RefCell<crate::value::JList>>, Throw> {
    match v {
        Value::List(l) => Ok(l),
        Value::Null => Err(npe()),
        o => Err(Throw::class_cast(format!(
            "{} is not a list",
            o.class_name()
        ))),
    }
}

/// Sorts `items` by natural order or by a comparator (stable, as Java's).
pub fn sort_values(vm: &mut Vm, items: &mut [Value], cmp: Option<&Value>) -> Result<(), Throw> {
    let mut err = None;
    let mut keyed: Vec<Value> = items.to_vec();
    // Merge sort by hand: the comparator may call back into the script.
    merge_sort(&mut keyed, &mut |a, b| {
        if err.is_some() {
            return Ordering::Equal;
        }
        let r = match cmp {
            Some(c) if !c.is_null() => invoke(vm, c, &[a.clone(), b.clone()])
                .and_then(|v| int_arg(&v))
                .map(|i| i.cmp(&0)),
            _ => compare::compare_to(a, b),
        };
        match r {
            Ok(o) => o,
            Err(e) => {
                err = Some(e);
                Ordering::Equal
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    items.clone_from_slice(&keyed);
    Ok(())
}

fn merge_sort(v: &mut [Value], cmp: &mut dyn FnMut(&Value, &Value) -> Ordering) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    if n < 8 {
        for i in 1..n {
            let mut j = i;
            while j > 0 && cmp(&v[j - 1], &v[j]) == Ordering::Greater {
                v.swap(j - 1, j);
                j -= 1;
            }
        }
        return;
    }
    let mid = n / 2;
    merge_sort(&mut v[..mid], cmp);
    merge_sort(&mut v[mid..], cmp);
    let left = v[..mid].to_vec();
    let right = v[mid..].to_vec();
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        if cmp(&right[j], &left[i]) == Ordering::Less {
            v[k] = right[j].clone();
            j += 1;
        } else {
            v[k] = left[i].clone();
            i += 1;
        }
        k += 1;
    }
    while i < left.len() {
        v[k] = left[i].clone();
        i += 1;
        k += 1;
    }
    while j < right.len() {
        v[k] = right[j].clone();
        j += 1;
        k += 1;
    }
}

fn min_max(vm: &mut Vm, a: &[Value], max: bool) -> Result<Value, Throw> {
    let items = elements(vm, &a[0])?;
    let mut best: Option<Value> = None;
    for x in items {
        best = Some(match best {
            None => x,
            Some(b) => {
                let o = match a.get(1) {
                    Some(c) if !c.is_null() => {
                        int_arg(&invoke(vm, c, &[x.clone(), b.clone()])?)?.cmp(&0)
                    }
                    _ => compare::compare_to(&x, &b)?,
                };
                if (max && o == Ordering::Greater) || (!max && o == Ordering::Less) {
                    x
                } else {
                    b
                }
            }
        });
    }
    best.ok_or_else(|| Throw::no_message("java.util.NoSuchElementException"))
}

pub fn collections_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("emptyList", 0) => |_vm, _a| Ok(Value::list(ListKind::Immutable, Vec::new())),
        ("emptyMap", 0) => |_vm, _a| Ok(Value::map(MapKind::Immutable, JMap::new(Order::Hash))),
        ("emptySet", 0) => |_vm, _a| new_set(SetKind::Immutable, Vec::new()),
        ("singletonList", 1) => |_vm, a| Ok(Value::list(ListKind::Immutable, vec![a[0].clone()])),
        ("singleton", 1) => |_vm, a| new_set(SetKind::Immutable, vec![a[0].clone()]),
        ("singletonMap", 2) => |_vm, a| {
            let mut m = JMap::new(Order::Hash);
            m.insert(a[0].clone(), a[1].clone());
            Ok(Value::map(MapKind::Immutable, m))
        },
        ("nCopies", 2) => |_vm, a| {
            let n = int_arg(&a[0])?;
            if n < 0 {
                return Err(Throw::illegal_arg(format!("List length = {n}")));
            }
            Ok(Value::list(
                ListKind::Immutable,
                vec![a[1].clone(); n as usize],
            ))
        },
        ("unmodifiableList", 1) => {
            |vm, a| Ok(Value::list(ListKind::Immutable, elements(vm, &a[0])?))
        }
        ("unmodifiableSet", 1) | ("unmodifiableCollection", 1) => |vm, a| match &a[0] {
            Value::List(_) => Ok(Value::list(ListKind::Immutable, elements(vm, &a[0])?)),
            _ => new_set(SetKind::Immutable, elements(vm, &a[0])?),
        },
        ("unmodifiableMap", 1) => |_vm, a| match &a[0] {
            Value::Map(m) => Ok(Value::map(MapKind::Immutable, m.borrow().map.clone())),
            Value::Null => Err(npe()),
            o => Err(Throw::class_cast(format!(
                "{} is not a map",
                o.class_name()
            ))),
        },
        ("sort", 1) | ("sort", 2) => |vm, a| {
            let l = list_ref(&a[0])?.clone();
            let mut items = l.borrow().items.clone();
            sort_values(vm, &mut items, a.get(1))?;
            l.borrow_mut().items = items;
            Ok(Value::Null)
        },
        ("reverse", 1) => |_vm, a| {
            list_ref(&a[0])?.borrow_mut().items.reverse();
            Ok(Value::Null)
        },
        ("max", 1) | ("max", 2) => |vm, a| min_max(vm, a, true),
        ("min", 1) | ("min", 2) => |vm, a| min_max(vm, a, false),
        ("frequency", 2) => |vm, a| {
            Ok(Value::Int(
                elements(vm, &a[0])?
                    .iter()
                    .filter(|x| x.java_equals(&a[1]))
                    .count() as i32,
            ))
        },
        ("swap", 3) => |_vm, a| {
            let l = list_ref(&a[0])?;
            let (i, j) = (int_arg(&a[1])?, int_arg(&a[2])?);
            let mut l = l.borrow_mut();
            let n = l.items.len();
            for x in [i, j] {
                if x < 0 || x as usize >= n {
                    return Err(Throw::index(
                        "java.lang.IndexOutOfBoundsException",
                        i64::from(x),
                        n,
                    ));
                }
            }
            l.items.swap(i as usize, j as usize);
            Ok(Value::Null)
        },
        ("fill", 2) => |_vm, a| {
            for x in list_ref(&a[0])?.borrow_mut().items.iter_mut() {
                *x = a[1].clone();
            }
            Ok(Value::Null)
        },
        ("addAll", 2) => |vm, a| {
            let items = elements(vm, &a[1])?;
            let mut changed = false;
            for x in items {
                changed |= add(&a[0], x)?;
            }
            Ok(Value::Bool(changed))
        },
        ("disjoint", 2) => |vm, a| {
            let xs = elements(vm, &a[0])?;
            for x in &xs {
                if contains(&a[1], x)? {
                    return Ok(Value::Bool(false));
                }
            }
            Ok(Value::Bool(true))
        },
        ("reverseOrder", 0) => |_vm, _a| Ok(api::functional::reverse_order()),
        ("reverseOrder", 1) => |_vm, a| Ok(api::functional::reversed(a[0].clone())),
        ("rotate", 2) => |_vm, a| {
            let l = list_ref(&a[0])?;
            let d = int_arg(&a[1])?;
            let mut l = l.borrow_mut();
            let n = l.items.len() as i32;
            if n > 0 {
                let d = d.rem_euclid(n) as usize;
                l.items.rotate_right(d);
            }
            Ok(Value::Null)
        },
        _ => return None,
    })
}
