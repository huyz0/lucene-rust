#![cfg(test)]
use super::*;
use crate::api::test_support::{call, i, s, t, text, vm_call};

fn al(xs: &[i32]) -> Value {
    Value::list(
        ListKind::ArrayList,
        xs.iter().map(|x| Value::Int(*x)).collect(),
    )
}

fn ll(xs: &[i32]) -> Value {
    Value::list(
        ListKind::LinkedList,
        xs.iter().map(|x| Value::Int(*x)).collect(),
    )
}

#[test]
fn indexed_reads_and_writes() {
    let l = al(&[10, 20, 30]);
    assert_eq!(t(&l, "get", &[i(1)]), "20");
    assert!(t(&l, "get", &[i(3)]).contains("IndexOutOfBounds"));
    assert!(t(&l, "get", &[i(-1)]).contains("IndexOutOfBounds"));
    assert_eq!(t(&l, "set", &[i(0), i(11)]), "10");
    assert_eq!(t(&l, "getLength", &[]), "3");
    assert_eq!(t(&l, "indexOf", &[i(30)]), "2");
    assert_eq!(t(&l, "indexOf", &[i(99)]), "-1");
    assert_eq!(t(&l, "lastIndexOf", &[i(20)]), "1");
    assert_eq!(t(&l, "subList", &[i(1), i(3)]), "[20, 30]");
    assert_eq!(
        t(&l, "subList", &[i(2), i(1)]),
        "java.lang.IllegalArgumentException: fromIndex(2) > toIndex(1)"
    );
    assert_eq!(t(&l, "clone", &[]), "[11, 20, 30]");
    assert_eq!(t(&l, "trimToSize", &[]), "null");
    assert_eq!(t(&l, "ensureCapacity", &[i(100)]), "null");
    assert_eq!(t(&l, "equals", &[al(&[11, 20, 30])]), "true");
    assert_eq!(t(&l, "hashCode", &[]), "41012");
}

#[test]
fn inserts_and_removals() {
    let l = al(&[1, 2, 3]);
    assert_eq!(t(&l, "add", &[i(0), i(0)]), "null");
    assert_eq!(t(&l, "add", &[i(4), i(4)]), "null");
    assert!(t(&l, "add", &[i(9), i(9)]).contains("IndexOutOfBounds"));
    assert_eq!(text(Ok(l.clone())), "[0, 1, 2, 3, 4]");
    assert_eq!(t(&l, "addAll", &[i(1), al(&[7, 8])]), "true");
    assert_eq!(t(&l, "addAll", &[i(1), al(&[])]), "false");
    assert!(t(&l, "addAll", &[i(99), al(&[1])]).contains("IndexOutOfBounds"));
    assert_eq!(text(Ok(l.clone())), "[0, 7, 8, 1, 2, 3, 4]");
    assert_eq!(t(&l, "remove", &[i(1)]), "7");
    assert!(t(&l, "remove", &[i(50)]).contains("IndexOutOfBounds"));
    // An object argument removes by equality; a `Long` is not an `Integer`.
    assert_eq!(t(&l, "remove", &[Value::Long(4)]), "false");
    assert_eq!(t(&l, "add", &[s("x")]), "true");
    assert_eq!(t(&l, "remove", &[s("x")]), "true");
    assert_eq!(t(&l, "remove", &[s("x")]), "false");
}

#[test]
fn immutable_lists_refuse_writes() {
    let imm = vm_call(static_lookup("of", 3).unwrap(), &[i(1), i(2), i(3)]).unwrap();
    assert_eq!(t(&imm, "get", &[i(2)]), "3");
    assert!(t(&imm, "set", &[i(0), i(9)]).contains("UnsupportedOperation"));
    assert!(t(&imm, "add", &[i(0), i(9)]).contains("UnsupportedOperation"));
    assert!(t(&imm, "addAll", &[i(0), al(&[1])]).contains("UnsupportedOperation"));
    assert!(t(&imm, "remove", &[i(0)]).contains("UnsupportedOperation"));
    let with_null = vm_call(static_lookup("of", 2).unwrap(), &[i(1), Value::Null]);
    assert!(text(with_null).contains("NullPointer"));
    let copy = vm_call(static_lookup("copyOf", 1).unwrap(), &[al(&[5, 6])]).unwrap();
    assert_eq!(text(Ok(copy.clone())), "[5, 6]");
    assert!(t(&copy, "set", &[i(0), i(1)]).contains("UnsupportedOperation"));
    assert!(static_lookup("nope", 1).is_none());
}

#[test]
fn deque_and_queue_methods_of_a_linked_list() {
    let q = ll(&[2, 3]);
    assert_eq!(t(&q, "addFirst", &[i(1)]), "null");
    assert_eq!(t(&q, "push", &[i(0)]), "null");
    assert_eq!(t(&q, "offerFirst", &[i(-1)]), "true");
    assert_eq!(t(&q, "addLast", &[i(4)]), "null");
    assert_eq!(t(&q, "offerLast", &[i(5)]), "true");
    assert_eq!(t(&q, "offer", &[i(6)]), "true");
    assert_eq!(text(Ok(q.clone())), "[-1, 0, 1, 2, 3, 4, 5, 6]");
    assert_eq!(t(&q, "getFirst", &[]), "-1");
    assert_eq!(t(&q, "element", &[]), "-1");
    assert_eq!(t(&q, "getLast", &[]), "6");
    assert_eq!(t(&q, "peek", &[]), "-1");
    assert_eq!(t(&q, "peekFirst", &[]), "-1");
    assert_eq!(t(&q, "peekLast", &[]), "6");
    assert_eq!(t(&q, "poll", &[]), "-1");
    assert_eq!(t(&q, "pollFirst", &[]), "0");
    assert_eq!(t(&q, "pollLast", &[]), "6");
    assert_eq!(t(&q, "pop", &[]), "1");
    assert_eq!(t(&q, "removeFirst", &[]), "2");
    assert_eq!(t(&q, "remove", &[]), "3");
    assert_eq!(t(&q, "removeLast", &[]), "5");
    assert_eq!(t(&q, "getFirst", &[]), "4");
    assert_eq!(t(&q, "pollLast", &[]), "4");
    let empty = ll(&[]);
    for m in [
        "getFirst",
        "element",
        "getLast",
        "pop",
        "removeFirst",
        "remove",
        "removeLast",
    ] {
        assert!(t(&empty, m, &[]).contains("NoSuchElement"), "{m}");
    }
    for m in [
        "peek",
        "peekFirst",
        "peekLast",
        "poll",
        "pollFirst",
        "pollLast",
    ] {
        assert_eq!(t(&empty, m, &[]), "null", "{m}");
    }
    let imm = vm_call(static_lookup("of", 1).unwrap(), &[i(1)]).unwrap();
    for m in [
        "addFirst",
        "offerFirst",
        "push",
        "addLast",
        "offerLast",
        "offer",
    ] {
        assert!(t(&imm, m, &[i(2)]).contains("UnsupportedOperation"), "{m}");
    }
}

#[test]
fn constructors_and_index_helpers() {
    let a = vm_call(constructor("java.util.ArrayList", 0).unwrap(), &[]).unwrap();
    assert_eq!(t(&a, "size", &[]), "0");
    let sized = vm_call(constructor("java.util.ArrayList", 1).unwrap(), &[i(8)]).unwrap();
    assert_eq!(t(&sized, "size", &[]), "0");
    let neg = vm_call(constructor("java.util.ArrayList", 1).unwrap(), &[i(-1)]);
    assert!(text(neg).contains("Illegal Capacity: -1"));
    let copy = vm_call(
        constructor("java.util.ArrayList", 1).unwrap(),
        &[al(&[1, 2])],
    )
    .unwrap();
    assert_eq!(text(Ok(copy)), "[1, 2]");
    let l = vm_call(constructor("java.util.LinkedList", 0).unwrap(), &[]).unwrap();
    assert_eq!(t(&l, "size", &[]), "0");
    let dq = vm_call(constructor("java.util.ArrayDeque", 1).unwrap(), &[i(4)]).unwrap();
    assert_eq!(t(&dq, "size", &[]), "0");
    let lc = vm_call(constructor("java.util.LinkedList", 1).unwrap(), &[al(&[3])]).unwrap();
    assert_eq!(text(Ok(lc)), "[3]");
    assert!(constructor("java.util.ArrayList", 2).is_none());
    let lit = vm_call(
        |vm: &mut Vm, a: &[Value]| new_array_list_of(vm, a),
        &[i(1), i(2)],
    )
    .unwrap();
    assert_eq!(text(Ok(lit)), "[1, 2]");
    let flip = |recv: &Value, ix: i32| text(vm_call(flip_index, &[recv.clone(), i(ix)]));
    assert_eq!(flip(&al(&[1, 2, 3]), -1), "2");
    assert_eq!(flip(&al(&[1, 2, 3]), 1), "1");
    assert!(flip(&Value::Null, -1).contains("NullPointerException"));
    assert_eq!(flip(&s("x"), -2), "-2");
    assert!(call(&Value::Null, "get", &[i(0)]).is_err());
    assert!(text(vm_call(get, &[Value::Null, i(0)])).contains("NullPointer"));
    assert!(text(vm_call(set, &[Value::Null, i(0), i(1)])).contains("NullPointer"));
}
