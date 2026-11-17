#![cfg(test)]
use super::*;
use crate::api::test_support::{i, s, text, vm_call};

fn ints(xs: &[i32]) -> Value {
    Value::array("int", xs.iter().map(|x| Value::Int(*x)).collect())
}

fn run(f: crate::vm::ApiFn, args: &[Value]) -> String {
    text(vm_call(f, args))
}

#[test]
fn length_load_and_store() {
    let a = ints(&[10, 20, 30]);
    assert_eq!(run(length, std::slice::from_ref(&a)), "3");
    assert_eq!(run(load, &[a.clone(), i(1)]), "20");
    assert!(run(load, &[a.clone(), i(3)]).contains("ArrayIndexOutOfBounds"));
    assert!(run(load, &[a.clone(), i(-1)]).contains("ArrayIndexOutOfBounds"));
    assert_eq!(run(store, &[a.clone(), i(0), i(99)]), "null");
    assert_eq!(run(load, &[a.clone(), i(0)]), "99");
    assert!(run(store, &[a.clone(), i(5), i(1)]).contains("ArrayIndexOutOfBounds"));
    assert!(run(store, &[a.clone(), i(0), Value::Null]).contains("NullPointer"));
    assert!(run(length, &[Value::Null]).contains("NullPointer"));
    assert!(run(length, &[s("x")]).contains("not an array"));
}

#[test]
fn stores_convert_to_the_component_type() {
    let l = Value::array("long", vec![Value::Long(0)]);
    run(store, &[l.clone(), i(0), i(7)]);
    assert!(matches!(&l, Value::Array(a) if matches!(a.borrow().items[0], Value::Long(7))));
    let objs = Value::array("java.lang.String", vec![Value::Null]);
    run(store, &[objs.clone(), i(0), Value::Null]);
    assert_eq!(run(load, &[objs, i(0)]), "null");
}

#[test]
fn negative_indexes_count_from_the_end() {
    let a = ints(&[1, 2, 3]);
    assert_eq!(run(flip_index, &[a.clone(), i(-1)]), "2");
    assert_eq!(run(flip_index, &[a.clone(), i(1)]), "1");
    assert!(run(flip_index, &[Value::Null, i(-1)]).contains("NullPointer"));
}

#[test]
fn defaults_literals_and_dimensions() {
    assert_eq!(default_element("int").java_to_string(), "0");
    assert_eq!(default_element("boolean").java_to_string(), "false");
    assert_eq!(default_element("double").java_to_string(), "0.0");
    assert!(default_element("java.lang.String").is_null());
    let lit = from_items("long", &[Value::Int(1), Value::Int(2)]).unwrap();
    assert!(matches!(&lit, Value::Array(a) if matches!(a.borrow().items[1], Value::Long(2))));
    assert!(from_items("int", &[Value::Null]).is_err());
    let grid = new_dims("[I", &[i(2), i(3)]).unwrap();
    assert_eq!(items(&grid).unwrap().len(), 2);
    assert_eq!(items(&items(&grid).unwrap()[0]).unwrap().len(), 3);
    let flat = new_dims("int", &[i(4)]).unwrap();
    assert_eq!(text(Ok(items(&flat).unwrap()[3].clone())), "0");
    assert!(text(new_dims("int", &[i(-1)])).contains("NegativeArraySize"));
    assert!(text(new_dims("int", &[i(i32::MAX)])).contains("OutOfMemory"));
    let text_grid = new_dims("[Ljava.lang.String;", &[i(1), i(2)]).unwrap();
    assert!(items(&items(&text_grid).unwrap()[0]).unwrap()[0].is_null());
    for (d, want) in [
        ("Z", "boolean"),
        ("B", "byte"),
        ("S", "short"),
        ("C", "char"),
        ("I", "int"),
        ("J", "long"),
        ("F", "float"),
        ("D", "double"),
        ("[I", "[I"),
        ("Ljava.lang.String;", "java.lang.String"),
    ] {
        assert_eq!(component_name(d), want);
    }
}

#[test]
fn arrays_helpers() {
    let nested = Value::array("def", vec![ints(&[1, 2]), ints(&[3])]);
    let f =
        |name: &str, args: &[Value]| text(vm_call(arrays_lookup(name, args.len()).unwrap(), args));
    assert_eq!(
        f("deepToString", std::slice::from_ref(&nested)),
        "[[1, 2], [3]]"
    );
    assert_eq!(f("deepEquals", &[nested.clone(), nested.clone()]), "true");
    let other = Value::array("def", vec![ints(&[1, 2]), ints(&[4])]);
    assert_eq!(f("deepEquals", &[nested.clone(), other]), "false");
    assert_eq!(f("deepEquals", &[ints(&[1]), ints(&[1, 2])]), "false");
    assert_eq!(
        f("deepHashCode", std::slice::from_ref(&nested)),
        f("deepHashCode", std::slice::from_ref(&nested))
    );
    assert_eq!(
        f("deepHashCode", &[Value::array("def", vec![Value::Null])]),
        "31"
    );
    assert_eq!(f("asList", &[ints(&[5, 6])]), "[5, 6]");
    assert!(arrays_lookup("sort", 1).is_none());
}
