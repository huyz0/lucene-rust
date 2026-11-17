#![cfg(test)]
//! The `Map` members that take a function (`compute*`, `merge`, `find`,
//! `groupBy`, ...) and `getByPath`'s errors.
use super::*;
use crate::api::functional::Composed;
use crate::api::test_support::{call, i, s, t, text, vm_call};

/// Two-argument function: `Integer.compare`-like `compareTo`.
fn cmp() -> Value {
    Value::object(Composed::NaturalOrder)
}

/// One-argument function returning its argument.
fn ident() -> Value {
    Value::object(Composed::Identity)
}

fn ints(kv: &[(i32, i32)]) -> Value {
    let m = vm_call(constructor("java.util.LinkedHashMap", 0).unwrap(), &[]).unwrap();
    for (k, v) in kv {
        call(&m, "put", &[i(*k), i(*v)]).unwrap();
    }
    m
}

fn flags() -> Value {
    let m = vm_call(constructor("java.util.LinkedHashMap", 0).unwrap(), &[]).unwrap();
    call(&m, "put", &[Value::Bool(true), i(1)]).unwrap();
    call(&m, "put", &[Value::Bool(false), i(2)]).unwrap();
    m
}

#[test]
fn compute_family_stores_what_the_function_returns() {
    let m = ints(&[(5, 3)]);
    assert_eq!(t(&m, "computeIfAbsent", &[i(9), ident()]), "9");
    assert_eq!(t(&m, "get", &[i(9)]), "9");
    assert_eq!(t(&m, "computeIfAbsent", &[i(5), ident()]), "3");
    assert_eq!(t(&m, "computeIfPresent", &[i(5), cmp()]), "1");
    assert_eq!(t(&m, "get", &[i(5)]), "1");
    assert_eq!(t(&m, "computeIfPresent", &[i(77), cmp()]), "null");
    assert_eq!(t(&m, "containsKey", &[i(77)]), "false");
    assert_eq!(t(&m, "compute", &[i(5), cmp()]), "1");
    assert_eq!(t(&m, "merge", &[i(5), i(9), cmp()]), "-1");
    assert_eq!(t(&m, "get", &[i(5)]), "-1");
    assert_eq!(t(&m, "merge", &[i(7), i(4), cmp()]), "4");
    assert_eq!(t(&m, "get", &[i(7)]), "4");
    assert_eq!(t(&m, "replaceAll", &[cmp()]), "null");
    // 5 vs -1, 9 vs 9, 7 vs 4.
    assert_eq!(text(Ok(m)), "{5=1, 9=0, 7=1}");
}

#[test]
fn predicates_over_entries() {
    let m = flags();
    assert_eq!(t(&m, "find", &[ident()]), "true=1");
    assert_eq!(t(&ints(&[]), "find", &[ident()]), "null");
    assert_eq!(t(&m, "findAll", &[ident()]), "{true=1}");
    assert_eq!(t(&m, "count", &[ident()]), "1");
    assert_eq!(t(&m, "every", &[ident()]), "false");
    assert_eq!(t(&ints(&[]), "every", &[ident()]), "true");
    assert_eq!(text(call(&m, "each", &[ident()])), "{true=1, false=2}");
    assert_eq!(t(&m, "forEach", &[ident()]), "null");
    assert_eq!(t(&m, "collect", &[ident()]), "[true, false]");
    let into = call(
        &vm_call(
            crate::api::list::constructor("java.util.ArrayList", 0).unwrap(),
            &[],
        )
        .unwrap(),
        "size",
        &[],
    )
    .unwrap();
    assert_eq!(text(Ok(into)), "0");
    assert!(t(&m, "count", &[s("not a function")]).contains("Exception"));
    assert!(t(&ints(&[(1, 1)]), "count", &[ident()]).contains("ClassCast"));
}

#[test]
fn group_by_groups_entries_under_the_functions_value() {
    let m = flags();
    // `Identity` over (key, value) returns the key: each entry is its own group.
    assert_eq!(
        t(&m, "groupBy", &[ident()]),
        "{false={false=2}, true={true=1}}"
    );
}

#[test]
fn get_by_path_names_what_is_wrong() {
    let inner = ints(&[(1, 2)]);
    let m = vm_call(constructor("java.util.HashMap", 0).unwrap(), &[]).unwrap();
    call(&m, "put", &[s("a"), inner]).unwrap();
    let list = vm_call(
        crate::api::list::constructor("java.util.ArrayList", 0).unwrap(),
        &[],
    )
    .unwrap();
    call(&list, "add", &[s("x")]).unwrap();
    call(&m, "put", &[s("l"), list]).unwrap();
    assert_eq!(t(&m, "getByPath", &[s("l.0")]), "x");
    assert_eq!(t(&m, "getByPath", &[s("l.5"), s("d")]), "d");
    assert!(t(&m, "getByPath", &[s("l.1")]).contains("IndexOutOfBounds"));
    assert!(t(&m, "getByPath", &[s("l.zz")]).contains("Could not parse [zz]"));
    assert!(t(&m, "getByPath", &[s("")]).contains("Missing path"));
    assert!(t(&m, "getByPath", &[s("a.")]).contains("Trailing '.' in path [a.]"));
    assert!(t(&m, "getByPath", &[s("a..b")]).contains("Extra '.' in path [a..b] at index [1]"));
    assert!(t(&m, "getByPath", &[s("q.r")]).contains("Container does not have [q]"));
    assert!(t(&m, "getByPath", &[s("q")]).contains("Could not find value at path [q]"));
    assert_eq!(t(&m, "getByPath", &[s("q"), i(7)]), "7");
    assert!(t(&m, "getByPath", &[s("l.0.z")]).contains("Non-container [java.lang.String] at [z]"));
}
