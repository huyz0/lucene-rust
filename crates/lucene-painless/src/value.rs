//! Runtime values: Java's primitives, boxed or not, strings, collections,
//! arrays and the host objects the allowlist exposes, with Java's
//! `equals`, `hashCode` and `toString`.
//!
//! A value belongs to one execution thread (`Rc`); what a compiled script
//! shares across threads holds no values, only constants.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::rc::Rc;

use crate::jmap::JMap;
use crate::number;

/// A Java string. Rust strings hold valid UTF-8, so a lone surrogate a
/// script builds from `char` arithmetic becomes U+FFFD; every index a script
/// sees (`length`, `charAt`, `substring`) is in UTF-16 units, as Java's.
pub type JStr = Rc<str>;

/// The kind of list a value was created as, which decides its class name
/// and whether it can change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListKind {
    ArrayList,
    LinkedList,
    /// `Collections.unmodifiableList`, `List.of`, `Arrays.asList` (fixed size).
    Immutable,
    FixedSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapKind {
    HashMap,
    LinkedHashMap,
    TreeMap,
    Immutable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetKind {
    HashSet,
    LinkedHashSet,
    TreeSet,
    Immutable,
}

#[derive(Debug, Clone)]
pub struct JList {
    pub kind: ListKind,
    pub items: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct JSet {
    pub kind: SetKind,
    pub map: JMap,
}

/// A Java array: its component type's binary name and its elements.
#[derive(Debug, Clone)]
pub struct JArray {
    pub component: Rc<str>,
    pub items: Vec<Value>,
}

/// A value a script holds.
#[derive(Debug, Clone, Default)]
pub enum Value {
    #[default]
    Null,
    Bool(bool),
    Byte(i8),
    Short(i16),
    Char(u16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Str(JStr),
    List(Rc<RefCell<JList>>),
    Map(Rc<RefCell<JMapValue>>),
    Set(Rc<RefCell<JSet>>),
    Array(Rc<RefCell<JArray>>),
    Object(Rc<dyn crate::object::JObject>),
}

#[derive(Debug, Clone)]
pub struct JMapValue {
    pub kind: MapKind,
    pub map: JMap,
}

impl Value {
    pub fn str(s: &str) -> Value {
        Value::Str(Rc::from(s))
    }

    pub fn string(s: String) -> Value {
        Value::Str(Rc::from(s))
    }

    pub fn list(kind: ListKind, items: Vec<Value>) -> Value {
        Value::List(Rc::new(RefCell::new(JList { kind, items })))
    }

    pub fn map(kind: MapKind, map: JMap) -> Value {
        Value::Map(Rc::new(RefCell::new(JMapValue { kind, map })))
    }

    pub fn set(kind: SetKind, map: JMap) -> Value {
        Value::Set(Rc::new(RefCell::new(JSet { kind, map })))
    }

    pub fn array(component: &str, items: Vec<Value>) -> Value {
        Value::Array(Rc::new(RefCell::new(JArray {
            component: Rc::from(component),
            items,
        })))
    }

    pub fn object(o: impl crate::object::JObject) -> Value {
        Value::Object(Rc::new(o))
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// The JVM binary name of the value's class, as `getClass().getName()`.
    pub fn class_name(&self) -> String {
        match self {
            Value::Null => "null".into(),
            Value::Bool(_) => "java.lang.Boolean".into(),
            Value::Byte(_) => "java.lang.Byte".into(),
            Value::Short(_) => "java.lang.Short".into(),
            Value::Char(_) => "java.lang.Character".into(),
            Value::Int(_) => "java.lang.Integer".into(),
            Value::Long(_) => "java.lang.Long".into(),
            Value::Float(_) => "java.lang.Float".into(),
            Value::Double(_) => "java.lang.Double".into(),
            Value::Str(_) => "java.lang.String".into(),
            Value::List(l) => match l.borrow().kind {
                ListKind::ArrayList => "java.util.ArrayList".into(),
                ListKind::LinkedList => "java.util.LinkedList".into(),
                ListKind::Immutable => "java.util.ImmutableCollections$ListN".into(),
                ListKind::FixedSize => "java.util.Arrays$ArrayList".into(),
            },
            Value::Map(m) => match m.borrow().kind {
                MapKind::HashMap => "java.util.HashMap".into(),
                MapKind::LinkedHashMap => "java.util.LinkedHashMap".into(),
                MapKind::TreeMap => "java.util.TreeMap".into(),
                MapKind::Immutable => "java.util.ImmutableCollections$MapN".into(),
            },
            Value::Set(s) => match s.borrow().kind {
                SetKind::HashSet => "java.util.HashSet".into(),
                SetKind::LinkedHashSet => "java.util.LinkedHashSet".into(),
                SetKind::TreeSet => "java.util.TreeSet".into(),
                SetKind::Immutable => "java.util.ImmutableCollections$SetN".into(),
            },
            Value::Array(a) => array_class_name(&a.borrow().component),
            Value::Object(o) => o.class_name().to_string(),
        }
    }

    /// `Object.equals`: Java's, so `Integer(1)` does not equal `Long(1)`.
    pub fn java_equals(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Null, _) | (_, Value::Null) => false,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Byte(a), Value::Byte(b)) => a == b,
            (Value::Short(a), Value::Short(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Long(a), Value::Long(b)) => a == b,
            // `Float.equals`/`Double.equals` compare bits: NaN equals NaN, 0.0 not -0.0.
            (Value::Float(a), Value::Float(b)) => {
                number::float_to_int_bits(*a) == number::float_to_int_bits(*b)
            }
            (Value::Double(a), Value::Double(b)) => {
                number::double_to_long_bits(*a) == number::double_to_long_bits(*b)
            }
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::List(a), Value::List(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (a, b) = (a.borrow(), b.borrow());
                a.items.len() == b.items.len()
                    && a.items.iter().zip(&b.items).all(|(x, y)| x.java_equals(y))
            }
            (Value::Map(a), Value::Map(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (a, b) = (a.borrow(), b.borrow());
                a.map.len() == b.map.len()
                    && a.map
                        .iter()
                        .all(|(k, v)| b.map.get(k).is_some_and(|w| v.java_equals(w)))
            }
            (Value::Set(a), Value::Set(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (a, b) = (a.borrow(), b.borrow());
                a.map.len() == b.map.len() && a.map.iter().all(|(k, _)| b.map.contains_key(k))
            }
            (Value::Array(a), Value::Array(b)) => Rc::ptr_eq(a, b),
            (Value::Object(a), Value::Object(b)) => Rc::ptr_eq(a, b) || a.java_equals(&**b),
            _ => false,
        }
    }

    /// `Object.hashCode`.
    pub fn java_hash(&self) -> i32 {
        match self {
            Value::Null => 0,
            Value::Bool(b) => {
                if *b {
                    1231
                } else {
                    1237
                }
            }
            Value::Byte(v) => i32::from(*v),
            Value::Short(v) => i32::from(*v),
            Value::Char(v) => i32::from(*v),
            Value::Int(v) => *v,
            Value::Long(v) => (*v ^ ((*v as u64) >> 32) as i64) as i32,
            Value::Float(v) => number::float_to_int_bits(*v),
            Value::Double(v) => {
                let b = number::double_to_long_bits(*v);
                (b ^ ((b as u64) >> 32) as i64) as i32
            }
            Value::Str(s) => string_hash(s),
            Value::List(l) => l
                .borrow()
                .items
                .iter()
                .fold(1i32, |h, v| h.wrapping_mul(31).wrapping_add(v.java_hash())),
            Value::Map(m) => m.borrow().map.iter().fold(0i32, |h, (k, v)| {
                h.wrapping_add(k.java_hash() ^ v.java_hash())
            }),
            Value::Set(s) => s
                .borrow()
                .map
                .iter()
                .fold(0i32, |h, (k, _)| h.wrapping_add(k.java_hash())),
            Value::Array(a) => (Rc::as_ptr(a) as usize as u64 as i64 >> 4) as i32,
            Value::Object(o) => o.java_hash(),
        }
    }

    /// `String.valueOf(value)`.
    pub fn java_to_string(&self) -> String {
        let mut s = String::new();
        self.write_java(&mut s, 0);
        s
    }

    pub(crate) fn write_java(&self, out: &mut String, depth: usize) {
        if depth > 64 {
            out.push_str("(this Collection)");
            return;
        }
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Byte(v) => {
                let _ = write!(out, "{v}");
            }
            Value::Short(v) => {
                let _ = write!(out, "{v}");
            }
            Value::Char(c) => out.push_str(&char_string(*c)),
            Value::Int(v) => {
                let _ = write!(out, "{v}");
            }
            Value::Long(v) => {
                let _ = write!(out, "{v}");
            }
            Value::Float(v) => out.push_str(&number::float_to_string(*v)),
            Value::Double(v) => out.push_str(&number::double_to_string(*v)),
            Value::Str(s) => out.push_str(s),
            Value::List(l) => {
                out.push('[');
                for (i, v) in l.borrow().items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_java(out, depth + 1);
                }
                out.push(']');
            }
            Value::Set(s) => {
                out.push('[');
                for (i, (v, _)) in s.borrow().map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_java(out, depth + 1);
                }
                out.push(']');
            }
            Value::Map(m) => {
                out.push('{');
                for (i, (k, v)) in m.borrow().map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    k.write_java(out, depth + 1);
                    out.push('=');
                    v.write_java(out, depth + 1);
                }
                out.push('}');
            }
            Value::Array(a) => {
                let a = a.borrow();
                let _ = write!(
                    out,
                    "{}@{:x}",
                    array_class_name(&a.component),
                    self.java_hash()
                );
            }
            Value::Object(o) => out.push_str(&o.java_to_string()),
        }
    }
}

/// A `char` as a one-character string (U+FFFD for a lone surrogate).
pub fn char_string(c: u16) -> String {
    char::from_u32(u32::from(c)).map_or_else(|| "\u{FFFD}".to_string(), |c| c.to_string())
}

/// `String.hashCode` over the UTF-16 units.
pub fn string_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// `Class.getName` of an array whose component type has binary name
/// `component`: `[I`, `[Ljava.lang.String;`, `[[D`.
pub fn array_class_name(component: &str) -> String {
    format!("[{}", descriptor(component))
}

/// The JVM descriptor of a binary class name, as array names nest it.
pub fn descriptor(name: &str) -> String {
    match name {
        "boolean" => "Z".into(),
        "byte" => "B".into(),
        "short" => "S".into(),
        "char" => "C".into(),
        "int" => "I".into(),
        "long" => "J".into(),
        "float" => "F".into(),
        "double" => "D".into(),
        n if n.starts_with('[') => n.to_string(),
        n => format!("L{n};"),
    }
}
