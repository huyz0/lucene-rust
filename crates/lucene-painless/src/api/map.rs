//! `Map` and its augmentations; `Map.Entry`.
//!
//! `keySet()`, `values()` and `entrySet()` return copies (Java's are live
//! views); `Map.Entry.setValue` writes through.

use std::cell::RefCell;
use std::rc::Rc;

use crate::api::{self, collection, int_arg, npe, str_arg};
use crate::compare;
use crate::jmap::{JMap, Order};
use crate::lambda::invoke;
use crate::object::JObject;
use crate::value::{JMapValue, ListKind, MapKind, SetKind, Value};
use crate::vm::{ApiFn, Throw, Vm};

fn order_of(kind: MapKind) -> Order {
    match kind {
        MapKind::HashMap | MapKind::Immutable => Order::Hash,
        MapKind::LinkedHashMap => Order::Insertion,
        MapKind::TreeMap => Order::Sorted,
    }
}

pub fn new_map(kind: MapKind) -> Value {
    Value::map(kind, JMap::new(order_of(kind)))
}

/// `[k: v, ...]`: a `HashMap`.
pub fn new_hash_map_of(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let mut m = JMap::new(Order::Hash);
    for kv in args.chunks(2) {
        m.insert(kv[0].clone(), kv[1].clone());
    }
    Ok(Value::map(MapKind::HashMap, m))
}

pub fn get_value(m: &JMapValue, k: &Value) -> Value {
    m.map.get(k).cloned().unwrap_or_default()
}

fn map_ref(v: &Value) -> Result<&Rc<RefCell<JMapValue>>, Throw> {
    match v {
        Value::Map(m) => Ok(m),
        Value::Null => Err(npe()),
        o => Err(Throw::class_cast(format!(
            "{} is not a map",
            o.class_name()
        ))),
    }
}

/// `map.put(k, v)`: the previous value.
pub fn put(_vm: &mut Vm, recv: &Value, k: &Value, v: &Value) -> Result<Value, Throw> {
    let m = map_ref(recv)?;
    let mut m = m.borrow_mut();
    match m.kind {
        MapKind::Immutable => {
            return Err(Throw::no_message("java.lang.UnsupportedOperationException"))
        }
        MapKind::TreeMap => collection::tree_key_check(&m.map, k)?,
        _ => {}
    }
    Ok(m.map.insert(k.clone(), v.clone()).unwrap_or_default())
}

pub fn constructor(class: &str, arity: usize) -> Option<ApiFn> {
    let kind = match class {
        "java.util.HashMap" | "java.util.Hashtable" => MapKind::HashMap,
        "java.util.LinkedHashMap" => MapKind::LinkedHashMap,
        _ => MapKind::TreeMap,
    };
    Some(match (kind, arity) {
        (MapKind::HashMap, 0) => |_vm, _a| Ok(new_map(MapKind::HashMap)),
        (MapKind::LinkedHashMap, 0) => |_vm, _a| Ok(new_map(MapKind::LinkedHashMap)),
        (MapKind::TreeMap, 0) => |_vm, _a| Ok(new_map(MapKind::TreeMap)),
        (MapKind::HashMap, 1) => |_vm, a| copy_or_capacity(MapKind::HashMap, &a[0]),
        (MapKind::LinkedHashMap, 1) => |_vm, a| copy_or_capacity(MapKind::LinkedHashMap, &a[0]),
        (MapKind::TreeMap, 1) => |_vm, a| copy_or_capacity(MapKind::TreeMap, &a[0]),
        _ => return None,
    })
}

fn copy_or_capacity(kind: MapKind, a: &Value) -> Result<Value, Throw> {
    match a {
        Value::Int(n) => {
            if *n < 0 {
                return Err(Throw::illegal_arg(format!("Illegal initial capacity: {n}")));
            }
            Ok(Value::map(
                kind,
                JMap::with_capacity(order_of(kind), *n as usize),
            ))
        }
        Value::Map(src) => {
            let src = src.borrow();
            let mut m = JMap::for_copy(order_of(kind), src.map.len());
            for (k, v) in src.map.iter() {
                if kind == MapKind::TreeMap {
                    collection::tree_key_check(&m, k)?;
                }
                m.insert(k.clone(), v.clone());
            }
            Ok(Value::map(kind, m))
        }
        Value::Null => Err(npe()),
        _ => Ok(new_map(kind)),
    }
}

pub fn static_lookup(name: &str, _arity: usize) -> Option<ApiFn> {
    Some(match name {
        "of" => |_vm, a| {
            let mut m = JMap::new(Order::Hash);
            for kv in a.chunks(2) {
                if kv[0].is_null() || kv[1].is_null() {
                    return Err(npe());
                }
                if m.insert(kv[0].clone(), kv[1].clone()).is_some() {
                    return Err(Throw::illegal_arg(format!(
                        "duplicate key: {}",
                        kv[0].java_to_string()
                    )));
                }
            }
            Ok(Value::map(MapKind::Immutable, m))
        },
        "copyOf" => |_vm, a| {
            Ok(Value::map(
                MapKind::Immutable,
                map_ref(&a[0])?.borrow().map.clone(),
            ))
        },
        "entry" => |_vm, a| {
            Ok(Value::object(MapEntry {
                key: a[0].clone(),
                value: RefCell::new(a[1].clone()),
                map: None,
            }))
        },
        _ => return None,
    })
}

/// A `Map.Entry`, writing `setValue` through to its map.
#[derive(Debug)]
pub struct MapEntry {
    pub key: Value,
    pub value: RefCell<Value>,
    pub map: Option<Rc<RefCell<JMapValue>>>,
}

impl JObject for MapEntry {
    fn class_name(&self) -> &str {
        "java.util.HashMap$Node"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn instance_of(&self, class: &str) -> bool {
        matches!(class, "java.util.Map$Entry" | "java.lang.Object")
    }

    fn java_equals(&self, other: &dyn JObject) -> bool {
        other.as_any().downcast_ref::<MapEntry>().is_some_and(|o| {
            o.key.java_equals(&self.key) && o.value.borrow().java_equals(&self.value.borrow())
        })
    }

    fn java_hash(&self) -> i32 {
        self.key.java_hash() ^ self.value.borrow().java_hash()
    }

    fn java_to_string(&self) -> String {
        format!(
            "{}={}",
            self.key.java_to_string(),
            self.value.borrow().java_to_string()
        )
    }

    fn call(
        &self,
        _vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(Ok(match (name, args.len()) {
            ("getKey", 0) => self.key.clone(),
            ("getValue", 0) => self.value.borrow().clone(),
            ("setValue", 1) => {
                if let Some(m) = &self.map {
                    if let Some(v) = m.borrow_mut().map.get_mut(&self.key) {
                        *v = args[0].clone();
                    }
                }
                std::mem::replace(&mut *self.value.borrow_mut(), args[0].clone())
            }
            _ => return None,
        }))
    }
}

fn entry(m: &Rc<RefCell<JMapValue>>, k: &Value, v: &Value) -> Value {
    Value::object(MapEntry {
        key: k.clone(),
        value: RefCell::new(v.clone()),
        map: Some(m.clone()),
    })
}

fn entries(m: &Rc<RefCell<JMapValue>>) -> Vec<(Value, Value)> {
    m.borrow()
        .map
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

fn truthy(v: &Value) -> Result<bool, Throw> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Null => Err(npe()),
        o => Err(Throw::cast_failure(&o.class_name(), "java.lang.Boolean")),
    }
}

fn check_mutable(m: &Rc<RefCell<JMapValue>>) -> Result<(), Throw> {
    if m.borrow().kind == MapKind::Immutable {
        Err(Throw::no_message("java.lang.UnsupportedOperationException"))
    } else {
        Ok(())
    }
}

fn key_set(m: &Rc<RefCell<JMapValue>>) -> Value {
    let m = m.borrow();
    let order = match m.kind {
        MapKind::TreeMap => Order::Sorted,
        MapKind::HashMap | MapKind::Immutable => Order::Hash,
        MapKind::LinkedHashMap => Order::Insertion,
    };
    let mut s = JMap::with_capacity(order, m.map.len().max(16));
    // Keep the map's own iteration order: a hash set of the same keys
    // iterates them the same way only with the map's table size.
    for k in m.map.keys() {
        s.insert(k, Value::Null);
    }
    let kind = match m.kind {
        MapKind::TreeMap => SetKind::TreeSet,
        MapKind::HashMap | MapKind::Immutable => SetKind::LinkedHashSet,
        MapKind::LinkedHashMap => SetKind::LinkedHashSet,
    };
    if kind == SetKind::LinkedHashSet {
        let mut s = JMap::new(Order::Insertion);
        for k in m.map.keys() {
            s.insert(k, Value::Null);
        }
        return Value::set(SetKind::LinkedHashSet, s);
    }
    Value::set(kind, s)
}

/// Methods of `Map` and its augmentations.
pub fn call(vm: &mut Vm, recv: &Value, name: &str, args: &[Value]) -> Option<Result<Value, Throw>> {
    let Value::Map(m) = recv else { return None };
    let m = m.clone();
    Some((|| -> Result<Value, Throw> {
        Ok(match (name, args.len()) {
            ("get", 1) => get_value(&m.borrow(), &args[0]),
            ("put", 2) => put(vm, recv, &args[0], &args[1])?,
            ("containsKey", 1) => Value::Bool(m.borrow().map.contains_key(&args[0])),
            ("containsValue", 1) => Value::Bool(m.borrow().map.values().iter().any(|v| v.java_equals(&args[0]))),
            ("size", 0) => Value::Int(m.borrow().map.len() as i32),
            ("isEmpty", 0) => Value::Bool(m.borrow().map.is_empty()),
            ("getOrDefault", 2) => m.borrow().map.get(&args[0]).cloned().unwrap_or_else(|| args[1].clone()),
            ("remove", 1) => {
                check_mutable(&m)?;
                m.borrow_mut().map.remove(&args[0]).unwrap_or_default()
            }
            ("remove", 2) => {
                check_mutable(&m)?;
                let hit = m.borrow().map.get(&args[0]).is_some_and(|v| v.java_equals(&args[1]));
                if hit {
                    m.borrow_mut().map.remove(&args[0]);
                }
                Value::Bool(hit)
            }
            ("clear", 0) => {
                check_mutable(&m)?;
                m.borrow_mut().map.clear();
                Value::Null
            }
            ("putAll", 1) => {
                let src = map_ref(&args[0])?.clone();
                for (k, v) in entries(&src) {
                    put(vm, recv, &k, &v)?;
                }
                Value::Null
            }
            ("putIfAbsent", 2) => {
                let cur = m.borrow().map.get(&args[0]).cloned().unwrap_or_default();
                if cur.is_null() {
                    put(vm, recv, &args[0], &args[1])?;
                }
                cur
            }
            ("replace", 2) => {
                if m.borrow().map.contains_key(&args[0]) {
                    put(vm, recv, &args[0], &args[1])?
                } else {
                    Value::Null
                }
            }
            ("replace", 3) => {
                let hit = m.borrow().map.get(&args[0]).is_some_and(|v| v.java_equals(&args[1]));
                if hit {
                    put(vm, recv, &args[0], &args[2])?;
                }
                Value::Bool(hit)
            }
            ("keySet", 0) => key_set(&m),
            ("values", 0) => Value::list(ListKind::ArrayList, m.borrow().map.values()),
            ("entrySet", 0) => {
                let es: Vec<Value> = entries(&m).iter().map(|(k, v)| entry(&m, k, v)).collect();
                let mut s = JMap::new(Order::Insertion);
                for e in es {
                    s.insert(e, Value::Null);
                }
                Value::set(SetKind::LinkedHashSet, s)
            }
            ("forEach", 1) | ("each", 1) => {
                for (k, v) in entries(&m) {
                    invoke(vm, &args[0], &[k, v])?;
                }
                if name == "each" {
                    recv.clone()
                } else {
                    Value::Null
                }
            }
            ("computeIfAbsent", 2) => {
                let cur = m.borrow().map.get(&args[0]).cloned().unwrap_or_default();
                if cur.is_null() {
                    let v = invoke(vm, &args[1], &[args[0].clone()])?;
                    if !v.is_null() {
                        put(vm, recv, &args[0], &v)?;
                    }
                    v
                } else {
                    cur
                }
            }
            ("computeIfPresent", 2) => {
                let cur = m.borrow().map.get(&args[0]).cloned().unwrap_or_default();
                if cur.is_null() {
                    Value::Null
                } else {
                    let v = invoke(vm, &args[1], &[args[0].clone(), cur])?;
                    if v.is_null() {
                        m.borrow_mut().map.remove(&args[0]);
                    } else {
                        put(vm, recv, &args[0], &v)?;
                    }
                    v
                }
            }
            ("compute", 2) => {
                let cur = m.borrow().map.get(&args[0]).cloned().unwrap_or_default();
                let v = invoke(vm, &args[1], &[args[0].clone(), cur])?;
                if v.is_null() {
                    m.borrow_mut().map.remove(&args[0]);
                } else {
                    put(vm, recv, &args[0], &v)?;
                }
                v
            }
            ("merge", 3) => {
                let cur = m.borrow().map.get(&args[0]).cloned().unwrap_or_default();
                let v = if cur.is_null() { args[1].clone() } else { invoke(vm, &args[2], &[cur, args[1].clone()])? };
                if v.is_null() {
                    m.borrow_mut().map.remove(&args[0]);
                } else {
                    put(vm, recv, &args[0], &v)?;
                }
                v
            }
            ("replaceAll", 1) => {
                for (k, v) in entries(&m) {
                    let nv = invoke(vm, &args[0], &[k.clone(), v])?;
                    if let Some(slot) = m.borrow_mut().map.get_mut(&k) {
                        *slot = nv;
                    }
                }
                Value::Null
            }
            ("equals", 1) => Value::Bool(recv.java_equals(&args[0])),
            ("hashCode", 0) => Value::Int(recv.java_hash()),
            ("getByPath", 1) | ("getByPath", 2) => get_by_path(vm, recv, args)?,
            ("collect", 1) => {
                let mut out = Vec::new();
                for (k, v) in entries(&m) {
                    out.push(invoke(vm, &args[0], &[k, v])?);
                }
                collection::new_list(out)
            }
            ("collect", 2) => {
                for (k, v) in entries(&m) {
                    let r = invoke(vm, &args[1], &[k, v])?;
                    collection::add(&args[0], r)?;
                }
                args[0].clone()
            }
            ("count", 1) => {
                let mut n = 0;
                for (k, v) in entries(&m) {
                    if truthy(&invoke(vm, &args[0], &[k, v])?)? {
                        n += 1;
                    }
                }
                Value::Int(n)
            }
            ("every", 1) => {
                for (k, v) in entries(&m) {
                    if !truthy(&invoke(vm, &args[0], &[k, v])?)? {
                        return Ok(Value::Bool(false));
                    }
                }
                Value::Bool(true)
            }
            ("find", 1) => {
                for (k, v) in entries(&m) {
                    if truthy(&invoke(vm, &args[0], &[k.clone(), v.clone()])?)? {
                        return Ok(entry(&m, &k, &v));
                    }
                }
                Value::Null
            }
            ("findAll", 1) => {
                let kind = m.borrow().kind;
                let out_kind = if kind == MapKind::Immutable { MapKind::HashMap } else { kind };
                let mut out = JMap::new(order_of(out_kind));
                for (k, v) in entries(&m) {
                    if truthy(&invoke(vm, &args[0], &[k.clone(), v.clone()])?)? {
                        out.insert(k, v);
                    }
                }
                Value::map(out_kind, out)
            }
            ("findResult", 1) | ("findResult", 2) => {
                let (default, f) = if args.len() == 2 { (args[0].clone(), &args[1]) } else { (Value::Null, &args[0]) };
                for (k, v) in entries(&m) {
                    let r = invoke(vm, f, &[k, v])?;
                    if !r.is_null() {
                        return Ok(r);
                    }
                }
                default
            }
            ("findResults", 1) => {
                let mut out = Vec::new();
                for (k, v) in entries(&m) {
                    let r = invoke(vm, &args[0], &[k, v])?;
                    if !r.is_null() {
                        out.push(r);
                    }
                }
                collection::new_list(out)
            }
            ("groupBy", 1) => {
                let kind = m.borrow().kind;
                let inner_kind = if kind == MapKind::Immutable { MapKind::HashMap } else { kind };
                let mut out = JMap::new(Order::Hash);
                for (k, v) in entries(&m) {
                    let g = invoke(vm, &args[0], &[k.clone(), v.clone()])?;
                    match out.get_mut(&g) {
                        Some(Value::Map(inner)) => {
                            inner.borrow_mut().map.insert(k, v);
                        }
                        _ => {
                            let mut inner = JMap::new(order_of(inner_kind));
                            inner.insert(k, v);
                            out.insert(g, Value::map(inner_kind, inner));
                        }
                    }
                }
                Value::map(MapKind::HashMap, out)
            }
            // SortedMap / NavigableMap, for a TreeMap.
            ("firstKey", 0) | ("lastKey", 0) => {
                let keys = m.borrow().map.keys();
                let k = if name == "firstKey" { keys.first() } else { keys.last() };
                k.cloned().ok_or_else(|| Throw::no_message("java.util.NoSuchElementException"))?
            }
            ("firstEntry", 0) | ("lastEntry", 0) => {
                let es = entries(&m);
                let e = if name == "firstEntry" { es.first() } else { es.last() };
                e.map_or(Value::Null, |(k, v)| entry(&m, k, v))
            }
            ("ceilingKey", 1) | ("floorKey", 1) | ("higherKey", 1) | ("lowerKey", 1) => {
                let keys = m.borrow().map.keys();
                collection::navigate(&keys, &args[0], name.trim_end_matches("Key")).cloned().unwrap_or_default()
            }
            ("ceilingEntry", 1) | ("floorEntry", 1) | ("higherEntry", 1) | ("lowerEntry", 1) => {
                let keys = m.borrow().map.keys();
                match collection::navigate(&keys, &args[0], name.trim_end_matches("Entry")).cloned() {
                    Some(k) => {
                        let v = get_value(&m.borrow(), &k);
                        entry(&m, &k, &v)
                    }
                    None => Value::Null,
                }
            }
            ("headMap", 1) | ("tailMap", 1) | ("headMap", 2) | ("tailMap", 2) => {
                let inclusive = args.get(1).map_or(Ok(name == "tailMap"), api::bool_arg)?;
                let mut out = JMap::new(Order::Sorted);
                for (k, v) in entries(&m) {
                    let o = compare::compare_to(&k, &args[0])?;
                    let keep = match (name, o) {
                        ("headMap", std::cmp::Ordering::Less) | ("tailMap", std::cmp::Ordering::Greater) => true,
                        (_, std::cmp::Ordering::Equal) => inclusive,
                        _ => false,
                    };
                    if keep {
                        out.insert(k, v);
                    }
                }
                Value::map(MapKind::TreeMap, out)
            }
            ("descendingMap", 0) => {
                let mut out = JMap::new(Order::Insertion);
                for (k, v) in entries(&m).into_iter().rev() {
                    out.insert(k, v);
                }
                Value::map(MapKind::LinkedHashMap, out)
            }
            ("pollFirstEntry", 0) | ("pollLastEntry", 0) => {
                let es = entries(&m);
                let e = if name == "pollFirstEntry" { es.first() } else { es.last() };
                match e {
                    Some((k, v)) => {
                        m.borrow_mut().map.remove(k);
                        Value::object(MapEntry { key: k.clone(), value: RefCell::new(v.clone()), map: None })
                    }
                    None => Value::Null,
                }
            }
            _ => return Err(Throw::new(api::UNSUPPORTED, "")),
        })
    })())
    .filter(|r| !matches!(r, Err(t) if t.class() == api::UNSUPPORTED && t.message().as_deref() == Some("")))
}

/// `getByPath(path)` / `getByPath(path, default)` on a list or map.
pub fn get_by_path(_vm: &mut Vm, recv: &Value, args: &[Value]) -> Result<Value, Throw> {
    let path = str_arg(&args[0])?;
    if path.is_empty() {
        return Err(Throw::illegal_arg("Missing path"));
    }
    if path.ends_with('.') {
        return Err(Throw::illegal_arg(format!("Trailing '.' in path [{path}]")));
    }
    let elements: Vec<&str> = path.split('.').collect();
    let default = args.get(1);
    let joined = elements.join(".");
    let mut cur = recv.clone();
    for (i, el) in elements.iter().enumerate() {
        if el.is_empty() {
            return Err(Throw::illegal_arg(format!(
                "Extra '.' in path [{joined}] at index [{i}]"
            )));
        }
        let next = match &cur {
            Value::Map(m) => {
                let k = Value::str(el);
                let m = m.borrow();
                m.map.contains_key(&k).then(|| get_value(&m, &k))
            }
            Value::List(l) => {
                let idx: i32 = el.parse().map_err(|_| {
                    Throw::illegal_arg(format!("Could not parse [{el}] as a int index into list at path [{joined}] and index [{i}]"))
                })?;
                let l = l.borrow();
                if idx >= 0 && (idx as usize) < l.items.len() {
                    Some(l.items[idx as usize].clone())
                } else if idx >= 0 && idx as usize == l.items.len() {
                    return Err(Throw::index(
                        "java.lang.IndexOutOfBoundsException",
                        i64::from(idx),
                        l.items.len(),
                    ));
                } else {
                    None
                }
            }
            other => {
                return Err(Throw::illegal_arg(format!(
                    "Non-container [{}] at [{el}], index [{i}] in path [{joined}]",
                    other.class_name()
                )))
            }
        };
        match next {
            Some(v) => cur = v,
            None => {
                if i == elements.len() - 1 {
                    return match default {
                        Some(d) => Ok(d.clone()),
                        None => Err(Throw::illegal_arg(format!(
                            "Could not find value at path [{path}]"
                        ))),
                    };
                }
                return Err(Throw::illegal_arg(format!(
                    "Container does not have [{el}], for non-terminal index [{i}] in path [{joined}]"
                )));
            }
        }
    }
    Ok(cur)
}

#[allow(dead_code)]
fn _int(v: &Value) -> Result<i32, Throw> {
    int_arg(v)
}
