#![cfg(test)]
use super::*;
use crate::api::test_support::{call, elems, i, s, t, text, vm_call};

fn set(class: &str, items: &[i32]) -> Value {
    let v = vm_call(set_constructor(class, 0).unwrap(), &[]).unwrap();
    for x in items {
        call(&v, "add", &[i(*x)]).unwrap();
    }
    v
}

fn list(items: &[i32]) -> Value {
    new_list(items.iter().map(|x| i(*x)).collect())
}

fn coll(name: &str, args: &[Value]) -> Result<Value, Throw> {
    vm_call(collections_lookup(name, args.len()).unwrap(), args)
}

#[test]
fn sets_keep_their_orders_and_reject_duplicates() {
    let h = set("java.util.LinkedHashSet", &[3, 1, 2, 3]);
    assert_eq!(text(Ok(h.clone())), "[3, 1, 2]");
    assert_eq!(t(&h, "size", &[]), "3");
    assert_eq!(t(&h, "add", &[i(1)]), "false");
    assert_eq!(t(&h, "add", &[i(9)]), "true");
    assert_eq!(t(&h, "contains", &[i(9)]), "true");
    assert_eq!(t(&h, "remove", &[i(9)]), "true");
    assert_eq!(t(&h, "remove", &[i(9)]), "false");
    assert_eq!(t(&h, "isEmpty", &[]), "false");
    let tree = set("java.util.TreeSet", &[5, 1, 3]);
    assert_eq!(text(Ok(tree.clone())), "[1, 3, 5]");
    assert_eq!(t(&tree, "first", &[]), "1");
    assert_eq!(t(&tree, "last", &[]), "5");
    assert_eq!(t(&tree, "ceiling", &[i(2)]), "3");
    assert_eq!(t(&tree, "floor", &[i(2)]), "1");
    assert_eq!(t(&tree, "higher", &[i(3)]), "5");
    assert_eq!(t(&tree, "lower", &[i(1)]), "null");
    assert_eq!(t(&tree, "pollFirst", &[]), "1");
    assert_eq!(t(&tree, "pollLast", &[]), "5");
    assert_eq!(text(Ok(tree.clone())), "[3]");
    assert_eq!(t(&tree, "clear", &[]), "null");
    assert!(t(&tree, "first", &[]).contains("NoSuchElement"));
    assert_eq!(t(&tree, "pollFirst", &[]), "null");
}

#[test]
fn bulk_set_operations() {
    let a = set("java.util.LinkedHashSet", &[1, 2, 3, 4]);
    assert_eq!(t(&a, "containsAll", &[list(&[1, 2])]), "true");
    assert_eq!(t(&a, "containsAll", &[list(&[1, 9])]), "false");
    assert_eq!(t(&a, "removeAll", &[list(&[1, 9])]), "true");
    assert_eq!(t(&a, "retainAll", &[list(&[2, 3])]), "true");
    assert_eq!(text(Ok(a.clone())), "[2, 3]");
    assert_eq!(t(&a, "addAll", &[list(&[3, 7])]), "true");
    assert_eq!(t(&a, "addAll", &[list(&[3, 7])]), "false");
    assert_eq!(
        t(&a, "equals", &[set("java.util.TreeSet", &[7, 3, 2])]),
        "true"
    );
    assert_eq!(t(&a, "hashCode", &[]), "12");
    assert_eq!(elems(call(&a, "toArray", &[])), "[2, 3, 7]");
    assert_eq!(t(&a, "join", &[s("|")]), "2|3|7");
    assert_eq!(t(&a, "sum", &[]), "12.0");
}

#[test]
fn set_constructors_and_statics() {
    let from_list = vm_call(
        set_constructor("java.util.HashSet", 1).unwrap(),
        &[list(&[1, 1, 2])],
    )
    .unwrap();
    assert_eq!(t(&from_list, "size", &[]), "2");
    let sized = vm_call(set_constructor("java.util.HashSet", 1).unwrap(), &[i(10)]).unwrap();
    assert_eq!(t(&sized, "size", &[]), "0");
    let linked = vm_call(
        set_constructor("java.util.LinkedHashSet", 1).unwrap(),
        &[list(&[3, 2, 3])],
    )
    .unwrap();
    assert_eq!(text(Ok(linked)), "[3, 2]");
    let tree = vm_call(
        set_constructor("java.util.TreeSet", 1).unwrap(),
        &[list(&[3, 2])],
    )
    .unwrap();
    assert_eq!(text(Ok(tree)), "[2, 3]");
    assert!(set_constructor("java.util.ArrayDeque", 0).is_none());
    let of = vm_call(set_static_lookup("of", 2).unwrap(), &[i(1), i(2)]).unwrap();
    assert!(t(&of, "add", &[i(3)]).contains("UnsupportedOperation"));
    assert!(t(&of, "clear", &[]).contains("UnsupportedOperation"));
    let copy = vm_call(set_static_lookup("copyOf", 1).unwrap(), &[list(&[4, 4])]).unwrap();
    assert_eq!(t(&copy, "size", &[]), "1");
    assert!(set_static_lookup("zip", 1).is_none());
}

#[test]
fn collection_queries_on_lists() {
    let l = list(&[4, 5, 6]);
    assert_eq!(t(&l, "size", &[]), "3");
    assert_eq!(t(&l, "contains", &[i(5)]), "true");
    assert_eq!(t(&l, "add", &[i(7)]), "true");
    assert_eq!(t(&l, "addAll", &[list(&[8, 9])]), "true");
    assert_eq!(t(&l, "containsAll", &[list(&[4, 9])]), "true");
    assert_eq!(t(&l, "removeAll", &[list(&[9])]), "true");
    assert_eq!(t(&l, "retainAll", &[list(&[4, 5, 6, 7])]), "true");
    assert_eq!(text(Ok(l.clone())), "[4, 5, 6, 7]");
    assert_eq!(t(&l, "join", &[s(",")]), "4,5,6,7");
    assert_eq!(t(&l, "sum", &[]), "22.0");
    assert_eq!(t(&l, "asList", &[]), "[4, 5, 6, 7]");
    assert_eq!(t(&l, "asCollection", &[]), "[4, 5, 6, 7]");
    assert_eq!(t(&l, "equals", &[list(&[4, 5, 6, 7])]), "true");
    assert_eq!(t(&l, "clear", &[]), "null");
    assert_eq!(t(&l, "isEmpty", &[]), "true");
}

#[test]
fn immutable_and_checked_mutation() {
    let imm = coll("singletonList", &[i(1)]).unwrap();
    assert!(t(&imm, "add", &[i(2)]).contains("UnsupportedOperation"));
    assert!(check_mutable(&imm).is_err());
    assert!(add(&imm, i(3)).is_err());
    let empty = coll("emptyList", &[]).unwrap();
    assert_eq!(t(&empty, "size", &[]), "0");
    assert!(contains(&Value::Int(1), &i(1)).is_err());
    assert!(check_mutable(&Value::Int(1)).is_ok() || check_mutable(&Value::Int(1)).is_err());
}

#[test]
fn collections_statics() {
    assert_eq!(text(coll("emptySet", &[])), "[]");
    assert_eq!(text(coll("emptyMap", &[])), "{}");
    assert_eq!(text(coll("singleton", &[i(4)])), "[4]");
    assert_eq!(text(coll("singletonMap", &[s("k"), i(1)])), "{k=1}");
    assert_eq!(text(coll("nCopies", &[i(3), s("x")])), "[x, x, x]");
    assert!(text(coll("nCopies", &[i(-1), s("x")])).contains("List length = -1"));
    assert_eq!(text(coll("unmodifiableList", &[list(&[1, 2])])), "[1, 2]");
    assert_eq!(
        text(coll(
            "unmodifiableSet",
            &[set("java.util.TreeSet", &[2, 1])]
        )),
        "[1, 2]"
    );
    assert_eq!(text(coll("unmodifiableCollection", &[list(&[1])])), "[1]");
    let m = vm_call(
        crate::api::map::constructor("java.util.TreeMap", 0).unwrap(),
        &[],
    )
    .unwrap();
    call(&m, "put", &[s("a"), i(1)]).unwrap();
    let um = coll("unmodifiableMap", &[m]).unwrap();
    assert!(t(&um, "put", &[s("b"), i(2)]).contains("UnsupportedOperation"));
}

#[test]
fn collections_algorithms_on_lists() {
    let l = list(&[3, 1, 2]);
    coll("sort", std::slice::from_ref(&l)).unwrap();
    assert_eq!(text(Ok(l.clone())), "[1, 2, 3]");
    coll("reverse", std::slice::from_ref(&l)).unwrap();
    assert_eq!(text(Ok(l.clone())), "[3, 2, 1]");
    assert_eq!(text(coll("max", std::slice::from_ref(&l))), "3");
    assert_eq!(text(coll("min", std::slice::from_ref(&l))), "1");
    coll("swap", &[l.clone(), i(0), i(2)]).unwrap();
    assert_eq!(text(Ok(l.clone())), "[1, 2, 3]");
    coll("rotate", &[l.clone(), i(1)]).unwrap();
    assert_eq!(text(Ok(l.clone())), "[3, 1, 2]");
    assert_eq!(text(coll("frequency", &[list(&[1, 2, 1]), i(1)])), "2");
    assert_eq!(text(coll("disjoint", &[list(&[1, 2]), list(&[3])])), "true");
    assert_eq!(
        text(coll("disjoint", &[list(&[1, 2]), list(&[2])])),
        "false"
    );
    coll("fill", &[l.clone(), i(0)]).unwrap();
    assert_eq!(text(Ok(l.clone())), "[0, 0, 0]");
    assert_eq!(
        text(coll(
            "addAll",
            &[l.clone(), Value::array("def", vec![i(5)])]
        )),
        "true"
    );
    assert!(text(coll("swap", &[l.clone(), i(0), i(99)])).contains("IndexOutOfBounds"));
    assert!(text(coll("max", &[list(&[])])).contains("NoSuchElement"));
    assert!(collections_lookup("nope", 1).is_none());
}

#[test]
fn navigate_picks_the_neighbouring_key() {
    let keys = [i(10), i(20), i(30)];
    assert_eq!(
        navigate(&keys, &i(20), "ceiling")
            .map(Value::java_to_string)
            .as_deref(),
        Some("20")
    );
    assert_eq!(
        navigate(&keys, &i(20), "higher")
            .map(Value::java_to_string)
            .as_deref(),
        Some("30")
    );
    assert_eq!(
        navigate(&keys, &i(20), "lower")
            .map(Value::java_to_string)
            .as_deref(),
        Some("10")
    );
    assert_eq!(
        navigate(&keys, &i(25), "floor")
            .map(Value::java_to_string)
            .as_deref(),
        Some("20")
    );
    assert!(navigate(&keys, &i(5), "floor").is_none());
    assert!(navigate(&keys, &i(35), "ceiling").is_none());
}
