#![cfg(test)]
use super::*;
use crate::api::test_support::{call, i, s, t, text, vm_call};

fn make(class: &str) -> Value {
    vm_call(constructor(class, 0).unwrap(), &[]).unwrap()
}

fn with(class: &str, kv: &[(&str, i32)]) -> Value {
    let m = make(class);
    for (k, v) in kv {
        call(&m, "put", &[s(k), i(*v)]).unwrap();
    }
    m
}

#[test]
fn put_get_and_remove() {
    let m = with("java.util.LinkedHashMap", &[("b", 2), ("a", 1)]);
    assert_eq!(text(Ok(m.clone())), "{b=2, a=1}");
    assert_eq!(t(&m, "size", &[]), "2");
    assert_eq!(t(&m, "get", &[s("a")]), "1");
    assert_eq!(t(&m, "get", &[s("zz")]), "null");
    assert_eq!(t(&m, "put", &[s("a"), i(10)]), "1");
    assert_eq!(t(&m, "containsKey", &[s("a")]), "true");
    assert_eq!(t(&m, "containsValue", &[i(2)]), "true");
    assert_eq!(t(&m, "containsValue", &[i(99)]), "false");
    assert_eq!(t(&m, "getOrDefault", &[s("q"), i(5)]), "5");
    assert_eq!(t(&m, "remove", &[s("b")]), "2");
    assert_eq!(t(&m, "remove", &[s("a"), i(0)]), "false");
    assert_eq!(t(&m, "remove", &[s("a"), i(10)]), "true");
    assert_eq!(t(&m, "isEmpty", &[]), "true");
    call(&m, "put", &[s("x"), i(1)]).unwrap();
    assert_eq!(t(&m, "clear", &[]), "null");
    assert_eq!(t(&m, "size", &[]), "0");
}

#[test]
fn conditional_updates() {
    let m = with("java.util.LinkedHashMap", &[("a", 1)]);
    assert_eq!(t(&m, "putIfAbsent", &[s("a"), i(9)]), "1");
    assert_eq!(t(&m, "putIfAbsent", &[s("b"), i(2)]), "null");
    assert_eq!(t(&m, "replace", &[s("b"), i(3)]), "2");
    assert_eq!(t(&m, "replace", &[s("zz"), i(3)]), "null");
    assert_eq!(t(&m, "replace", &[s("b"), i(3), i(4)]), "true");
    assert_eq!(t(&m, "replace", &[s("b"), i(3), i(5)]), "false");
    let other = with("java.util.LinkedHashMap", &[("c", 7)]);
    assert_eq!(t(&m, "putAll", &[other]), "null");
    assert_eq!(text(Ok(m.clone())), "{a=1, b=4, c=7}");
    assert_eq!(t(&m, "keySet", &[]), "[a, b, c]");
    assert_eq!(t(&m, "values", &[]), "[1, 4, 7]");
    assert_eq!(t(&m, "entrySet", &[]), "[a=1, b=4, c=7]");
}

#[test]
fn equality_and_hash_follow_contents() {
    let a = with("java.util.HashMap", &[("x", 1), ("y", 2)]);
    let b = with("java.util.TreeMap", &[("y", 2), ("x", 1)]);
    assert_eq!(t(&a, "equals", std::slice::from_ref(&b)), "true");
    assert_eq!(t(&a, "hashCode", &[]), t(&b, "hashCode", &[]));
    call(&b, "put", &[s("z"), i(3)]).unwrap();
    assert_eq!(t(&a, "equals", &[b]), "false");
}

#[test]
fn tree_map_navigation() {
    let m = with(
        "java.util.TreeMap",
        &[("b", 2), ("d", 4), ("a", 1), ("c", 3)],
    );
    assert_eq!(text(Ok(m.clone())), "{a=1, b=2, c=3, d=4}");
    assert_eq!(t(&m, "firstKey", &[]), "a");
    assert_eq!(t(&m, "lastKey", &[]), "d");
    assert_eq!(t(&m, "firstEntry", &[]), "a=1");
    assert_eq!(t(&m, "lastEntry", &[]), "d=4");
    assert_eq!(t(&m, "ceilingKey", &[s("bb")]), "c");
    assert_eq!(t(&m, "floorKey", &[s("bb")]), "b");
    assert_eq!(t(&m, "higherKey", &[s("b")]), "c");
    assert_eq!(t(&m, "lowerKey", &[s("b")]), "a");
    assert_eq!(t(&m, "lowerKey", &[s("a")]), "null");
    assert_eq!(t(&m, "ceilingEntry", &[s("c")]), "c=3");
    assert_eq!(t(&m, "floorEntry", &[s("c")]), "c=3");
    assert_eq!(t(&m, "higherEntry", &[s("c")]), "d=4");
    assert_eq!(t(&m, "lowerEntry", &[s("zz")]), "d=4");
    assert_eq!(t(&m, "higherEntry", &[s("d")]), "null");
    assert_eq!(t(&m, "headMap", &[s("c")]), "{a=1, b=2}");
    assert_eq!(
        t(&m, "headMap", &[s("c"), Value::Bool(true)]),
        "{a=1, b=2, c=3}"
    );
    assert_eq!(t(&m, "tailMap", &[s("c")]), "{c=3, d=4}");
    assert_eq!(t(&m, "tailMap", &[s("c"), Value::Bool(false)]), "{d=4}");
    assert_eq!(t(&m, "descendingMap", &[]), "{d=4, c=3, b=2, a=1}");
    assert_eq!(t(&m, "pollFirstEntry", &[]), "a=1");
    assert_eq!(t(&m, "pollLastEntry", &[]), "d=4");
    assert_eq!(t(&m, "size", &[]), "2");
    let empty = make("java.util.TreeMap");
    assert!(t(&empty, "firstKey", &[]).contains("NoSuchElement"));
    assert_eq!(t(&empty, "firstEntry", &[]), "null");
    assert_eq!(t(&empty, "pollFirstEntry", &[]), "null");
}

#[test]
fn immutable_maps_refuse_writes() {
    let m = vm_call(
        static_lookup("of", 4).unwrap(),
        &[s("a"), i(1), s("b"), i(2)],
    )
    .unwrap();
    assert_eq!(t(&m, "get", &[s("b")]), "2");
    assert!(t(&m, "put", &[s("c"), i(3)]).contains("UnsupportedOperation"));
    assert!(t(&m, "remove", &[s("a")]).contains("UnsupportedOperation"));
    assert!(t(&m, "clear", &[]).contains("UnsupportedOperation"));
    let dup = vm_call(
        static_lookup("of", 4).unwrap(),
        &[s("a"), i(1), s("a"), i(2)],
    );
    assert!(text(dup).contains("duplicate key: a"));
    let nul = vm_call(static_lookup("of", 2).unwrap(), &[s("a"), Value::Null]);
    assert!(text(nul).contains("NullPointer"));
    let copy = vm_call(
        static_lookup("copyOf", 1).unwrap(),
        &[with("java.util.TreeMap", &[("k", 1)])],
    )
    .unwrap();
    assert_eq!(text(Ok(copy)), "{k=1}");
    let e = vm_call(static_lookup("entry", 2).unwrap(), &[s("k"), i(9)]).unwrap();
    assert_eq!(t(&e, "getKey", &[]), "k");
    assert_eq!(t(&e, "getValue", &[]), "9");
}

#[test]
fn constructors_copy_or_size() {
    let src = with("java.util.LinkedHashMap", &[("a", 1)]);
    let copy = vm_call(
        constructor("java.util.TreeMap", 1).unwrap(),
        std::slice::from_ref(&src),
    )
    .unwrap();
    assert_eq!(text(Ok(copy)), "{a=1}");
    let sized = vm_call(constructor("java.util.HashMap", 1).unwrap(), &[i(16)]).unwrap();
    assert_eq!(t(&sized, "size", &[]), "0");
    let neg = vm_call(constructor("java.util.HashMap", 1).unwrap(), &[i(-1)]);
    assert!(text(neg).contains("Illegal initial capacity: -1"));
    let nul = vm_call(
        constructor("java.util.LinkedHashMap", 1).unwrap(),
        &[Value::Null],
    );
    assert!(text(nul).contains("NullPointer"));
    assert!(constructor("java.util.HashMap", 3).is_none());
    assert!(t(&make("java.util.HashMap"), "nope", &[]).contains("not found"));
}

#[test]
fn get_by_path_walks_nested_maps_and_lists() {
    let inner = with("java.util.LinkedHashMap", &[("n", 5)]);
    let outer = make("java.util.LinkedHashMap");
    call(&outer, "put", &[s("in"), inner]).unwrap();
    assert_eq!(t(&outer, "getByPath", &[s("in.n")]), "5");
    assert_eq!(t(&outer, "getByPath", &[s("in.zz"), i(0)]), "0");
    assert!(
        t(&outer, "getByPath", &[s("in.zz")]).contains("IllegalArgument")
            || t(&outer, "getByPath", &[s("in.zz")]).contains("null")
    );
}
