//! The runtime library: the classes of Painless's allowlist, by binary
//! name, with Java's behaviour.
//!
//! Instance methods dispatch on the receiver's runtime value
//! ([`dynamic::call`]), so a typed call and a `def` call reach the same
//! code; static methods and constructors are found by class, name and
//! arity when the script is compiled.

pub mod array;
pub mod collection;
pub mod doc;
pub mod dynamic;
pub mod functional;
pub mod iter;
pub mod list;
pub mod map;
pub mod math;
pub mod number;
pub mod objects;
pub mod regex;
pub mod score;
pub mod string;

use std::rc::Rc;

use crate::ir;
use crate::types::{as_f64, as_i64};
use crate::value::Value;
use crate::vm::{ApiFn, Native, Throw, Vm};
use crate::Result;

/// The class of the exception the runtime throws when a script reaches a
/// library method it does not implement.
pub const UNSUPPORTED: &str = "org.lucenerust.painless.Unsupported";

/// Whether `class` is a class whose static methods Painless exposes as
/// instance methods of another class (`Augmentation`).
pub fn is_augmentation(class: &str) -> bool {
    class.ends_with("Augmentation")
}

/// A static method.
pub fn static_method(class: &str, name: &str, arity: usize) -> Option<ApiFn> {
    match class {
        "java.lang.Math" | "java.lang.StrictMath" => math::lookup(name, arity),
        "java.lang.Integer"
        | "java.lang.Long"
        | "java.lang.Short"
        | "java.lang.Byte"
        | "java.lang.Double"
        | "java.lang.Float"
        | "java.lang.Boolean"
        | "java.lang.Character" => number::static_lookup(class, name, arity),
        "java.lang.String" => string::static_lookup(name, arity),
        "java.lang.CharSequence" => string::static_lookup(name, arity),
        "java.util.Objects" => objects::objects_lookup(name, arity),
        "java.util.Arrays" => array::arrays_lookup(name, arity),
        "java.util.Collections" => collection::collections_lookup(name, arity),
        "java.util.List" => list::static_lookup(name, arity),
        "java.util.Map" => map::static_lookup(name, arity),
        "java.util.Set" => collection::set_static_lookup(name, arity),
        "java.util.regex.Pattern" => regex::static_lookup(name, arity),
        "java.util.Comparator" => functional::comparator_static(name, arity),
        "java.util.function.Function" | "java.util.function.UnaryOperator" => {
            functional::function_static(name, arity)
        }
        "java.lang.System" => objects::system_lookup(name, arity),
        score::UTILS => score::static_lookup(name, arity),
        _ => None,
    }
}

/// A constructor.
pub fn constructor(class: &str, arity: usize) -> Option<ApiFn> {
    match class {
        "java.util.ArrayList"
        | "java.util.LinkedList"
        | "java.util.ArrayDeque"
        | "java.util.Stack"
        | "java.util.Vector" => list::constructor(class, arity),
        "java.util.HashMap"
        | "java.util.LinkedHashMap"
        | "java.util.TreeMap"
        | "java.util.Hashtable" => map::constructor(class, arity),
        "java.util.HashSet" | "java.util.LinkedHashSet" | "java.util.TreeSet" => {
            collection::set_constructor(class, arity)
        }
        "java.lang.StringBuilder" | "java.lang.StringBuffer" => string::builder_constructor(arity),
        "java.lang.String" => string::string_constructor(arity),
        _ => None,
    }
}

/// A static field not folded into a constant.
pub fn static_field(class: &str, name: &str) -> Option<Value> {
    Some(match (class, name) {
        ("java.lang.Math", "PI") => Value::Double(std::f64::consts::PI),
        ("java.lang.Math", "E") => Value::Double(std::f64::consts::E),
        ("java.lang.Integer", "MAX_VALUE") => Value::Int(i32::MAX),
        ("java.lang.Integer", "MIN_VALUE") => Value::Int(i32::MIN),
        ("java.lang.Long", "MAX_VALUE") => Value::Long(i64::MAX),
        ("java.lang.Long", "MIN_VALUE") => Value::Long(i64::MIN),
        ("java.lang.Double", "MAX_VALUE") => Value::Double(f64::MAX),
        ("java.lang.Double", "MIN_VALUE") => Value::Double(f64::from_bits(1)),
        ("java.lang.Double", "POSITIVE_INFINITY") => Value::Double(f64::INFINITY),
        ("java.lang.Double", "NEGATIVE_INFINITY") => Value::Double(f64::NEG_INFINITY),
        ("java.lang.Double", "NaN") => Value::Double(f64::NAN),
        ("java.lang.Float", "MAX_VALUE") => Value::Float(f32::MAX),
        ("java.lang.Float", "MIN_VALUE") => Value::Float(f32::from_bits(1)),
        ("java.lang.Float", "NaN") => Value::Float(f32::NAN),
        ("java.lang.Float", "POSITIVE_INFINITY") => Value::Float(f32::INFINITY),
        ("java.lang.Float", "NEGATIVE_INFINITY") => Value::Float(f32::NEG_INFINITY),
        ("java.lang.Short", "MAX_VALUE") => Value::Short(i16::MAX),
        ("java.lang.Short", "MIN_VALUE") => Value::Short(i16::MIN),
        ("java.lang.Byte", "MAX_VALUE") => Value::Byte(i8::MAX),
        ("java.lang.Byte", "MIN_VALUE") => Value::Byte(i8::MIN),
        ("java.lang.Character", "MAX_VALUE") => Value::Char(u16::MAX),
        ("java.lang.Character", "MIN_VALUE") => Value::Char(0),
        ("java.lang.Boolean", "TRUE") => Value::Bool(true),
        ("java.lang.Boolean", "FALSE") => Value::Bool(false),
        ("java.util.Collections", "EMPTY_LIST") => {
            Value::list(crate::value::ListKind::Immutable, Vec::new())
        }
        ("java.util.Collections", "EMPTY_MAP") => Value::map(
            crate::value::MapKind::Immutable,
            crate::jmap::JMap::new(crate::jmap::Order::Hash),
        ),
        _ => return None,
    })
}

/// A class binding: constructed once per script instance, then called.
#[derive(Clone, Copy)]
pub struct ClassBinding {
    pub construct: ApiFn,
    pub call: fn(&mut Vm, &Value, &[Value]) -> std::result::Result<Value, Throw>,
}

/// The class bindings of the base allowlist and the score context.
pub fn class_binding(class: &str, _method: &str) -> Option<ClassBinding> {
    score::class_binding(class)
}

/// The binary name of a class a `def` reference names canonically
/// (`String`, `Integer`, `ArrayList`).
pub fn resolve_class(name: &str) -> String {
    if name.contains('.') {
        return name.to_string();
    }
    match name {
        "Math" | "String" | "Integer" | "Long" | "Short" | "Byte" | "Double" | "Float"
        | "Boolean" | "Character" | "Object" | "StringBuilder" | "CharSequence" | "Comparable"
        | "Number" | "StrictMath" | "System" => format!("java.lang.{name}"),
        "ArrayList" | "LinkedList" | "HashMap" | "TreeMap" | "LinkedHashMap" | "HashSet"
        | "TreeSet" | "LinkedHashSet" | "List" | "Map" | "Set" | "Collections" | "Arrays"
        | "Objects" | "Collection" | "Comparator" | "Optional" | "Iterator" => {
            format!("java.util.{name}")
        }
        other => other.to_string(),
    }
}

/// A compiled instance call of `m`: dispatched on the receiver's value.
pub fn method_call(m: &ir::Method) -> Result<Native> {
    let name: Rc<str> = Rc::from(m.name.as_str());
    // `Collection.remove(Object)` and `List.remove(int)` share a name.
    if &*name == "remove" && m.params.first().is_some_and(|p| p != "int") {
        return Ok(Rc::new(|vm: &mut Vm, args: &[Value]| {
            collection::remove_object(vm, &args[0], &args[1])
        }));
    }
    Ok(Rc::new(move |vm: &mut Vm, args: &[Value]| {
        dynamic::call(vm, &args[0], &name, &args[1..])
    }))
}

// Argument helpers.

pub(crate) fn npe() -> Throw {
    Throw::no_message("java.lang.NullPointerException")
}

pub(crate) fn int_arg(v: &Value) -> std::result::Result<i32, Throw> {
    match v {
        Value::Null => Err(npe()),
        Value::Str(_) => Err(Throw::new(
            "java.lang.invoke.WrongMethodTypeException",
            "cannot convert a String argument to int",
        )),
        Value::Int(i) => Ok(*i),
        other => as_i64(other)
            .map(|x| x as i32)
            .ok_or_else(|| Throw::class_cast(format!("{} is not an int", other.class_name()))),
    }
}

pub(crate) fn long_arg(v: &Value) -> std::result::Result<i64, Throw> {
    match v {
        Value::Null => Err(npe()),
        other => as_i64(other)
            .ok_or_else(|| Throw::class_cast(format!("{} is not a long", other.class_name()))),
    }
}

pub(crate) fn double_arg(v: &Value) -> std::result::Result<f64, Throw> {
    match v {
        Value::Null => Err(npe()),
        other => as_f64(other)
            .ok_or_else(|| Throw::class_cast(format!("{} is not a double", other.class_name()))),
    }
}

pub(crate) fn bool_arg(v: &Value) -> std::result::Result<bool, Throw> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Null => Err(npe()),
        other => Err(Throw::class_cast(format!(
            "{} is not a boolean",
            other.class_name()
        ))),
    }
}

pub(crate) fn str_arg(v: &Value) -> std::result::Result<Rc<str>, Throw> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        Value::Null => Err(npe()),
        Value::Object(o) if o.as_any().is::<string::JStringBuilder>() => {
            Ok(Rc::from(o.java_to_string().as_str()))
        }
        other => Err(Throw::cast_failure(&other.class_name(), "java.lang.String")),
    }
}
