//! Painless types as the IR names them (JVM binary names), Java's class
//! hierarchy for the classes [`Value`] models, and the numeric conversions.

use crate::value::{ListKind, MapKind, SetKind, Value};

pub const DEF: &str = "org.opensearch.painless.lookup.def";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Prim {
    Bool,
    Byte,
    Short,
    Char,
    Int,
    Long,
    Float,
    Double,
}

impl Prim {
    pub fn name(self) -> &'static str {
        match self {
            Prim::Bool => "boolean",
            Prim::Byte => "byte",
            Prim::Short => "short",
            Prim::Char => "char",
            Prim::Int => "int",
            Prim::Long => "long",
            Prim::Float => "float",
            Prim::Double => "double",
        }
    }

    pub fn boxed(self) -> &'static str {
        match self {
            Prim::Bool => "java.lang.Boolean",
            Prim::Byte => "java.lang.Byte",
            Prim::Short => "java.lang.Short",
            Prim::Char => "java.lang.Character",
            Prim::Int => "java.lang.Integer",
            Prim::Long => "java.lang.Long",
            Prim::Float => "java.lang.Float",
            Prim::Double => "java.lang.Double",
        }
    }
}

/// The primitive a type name denotes.
pub fn prim(ty: &str) -> Option<Prim> {
    Some(match ty {
        "boolean" => Prim::Bool,
        "byte" => Prim::Byte,
        "short" => Prim::Short,
        "char" => Prim::Char,
        "int" => Prim::Int,
        "long" => Prim::Long,
        "float" => Prim::Float,
        "double" => Prim::Double,
        _ => return None,
    })
}

/// The primitive a box class unboxes to.
pub fn unboxed(ty: &str) -> Option<Prim> {
    Some(match ty {
        "java.lang.Boolean" => Prim::Bool,
        "java.lang.Byte" => Prim::Byte,
        "java.lang.Short" => Prim::Short,
        "java.lang.Character" => Prim::Char,
        "java.lang.Integer" => Prim::Int,
        "java.lang.Long" => Prim::Long,
        "java.lang.Float" => Prim::Float,
        "java.lang.Double" => Prim::Double,
        _ => return None,
    })
}

/// JVM stack words a value of type `ty` takes.
pub fn words(ty: &str) -> usize {
    match ty {
        "void" => 0,
        "long" | "double" => 2,
        _ => 1,
    }
}

/// The primitive kind of a boxed or primitive value.
pub fn prim_of(v: &Value) -> Option<Prim> {
    Some(match v {
        Value::Bool(_) => Prim::Bool,
        Value::Byte(_) => Prim::Byte,
        Value::Short(_) => Prim::Short,
        Value::Char(_) => Prim::Char,
        Value::Int(_) => Prim::Int,
        Value::Long(_) => Prim::Long,
        Value::Float(_) => Prim::Float,
        Value::Double(_) => Prim::Double,
        _ => return None,
    })
}

/// A numeric (or `char`) value as `i64`, Java's widening/narrowing to long.
pub fn as_i64(v: &Value) -> Option<i64> {
    Some(match v {
        Value::Byte(x) => i64::from(*x),
        Value::Short(x) => i64::from(*x),
        Value::Char(x) => i64::from(*x),
        Value::Int(x) => i64::from(*x),
        Value::Long(x) => *x,
        Value::Float(x) => *x as i64,
        Value::Double(x) => *x as i64,
        _ => return None,
    })
}

pub fn as_f64(v: &Value) -> Option<f64> {
    Some(match v {
        Value::Byte(x) => f64::from(*x),
        Value::Short(x) => f64::from(*x),
        Value::Char(x) => f64::from(*x),
        Value::Int(x) => f64::from(*x),
        Value::Long(x) => *x as f64,
        Value::Float(x) => f64::from(*x),
        Value::Double(x) => *x,
        _ => return None,
    })
}

pub fn as_f32(v: &Value) -> Option<f32> {
    Some(match v {
        Value::Long(x) => *x as f32,
        Value::Float(x) => *x,
        Value::Double(x) => *x as f32,
        other => as_i64(other)? as f32,
    })
}

/// Java's primitive conversion of a numeric/char value to `to` (`(int) 3.7`,
/// `(byte) 300`, `(char) 65`). `None` for a boolean or reference.
pub fn convert(v: &Value, to: Prim) -> Option<Value> {
    if let Value::Bool(b) = v {
        return (to == Prim::Bool).then_some(Value::Bool(*b));
    }
    prim_of(v)?;
    Some(match to {
        Prim::Bool => return None,
        Prim::Byte => Value::Byte(int_of(v) as i8),
        Prim::Short => Value::Short(int_of(v) as i16),
        Prim::Char => Value::Char(int_of(v) as u16),
        Prim::Int => Value::Int(int_of(v)),
        Prim::Long => Value::Long(match v {
            Value::Float(x) => *x as i64,
            Value::Double(x) => *x as i64,
            _ => as_i64(v)?,
        }),
        Prim::Float => Value::Float(as_f32(v)?),
        Prim::Double => Value::Double(as_f64(v)?),
    })
}

/// `(int) v`: `l2i` truncates, `f2i`/`d2i` saturate.
fn int_of(v: &Value) -> i32 {
    match v {
        Value::Float(x) => *x as i32,
        Value::Double(x) => *x as i32,
        Value::Long(x) => *x as i32,
        _ => as_i64(v).unwrap_or(0) as i32,
    }
}

pub fn is_number(v: &Value) -> bool {
    matches!(
        v,
        Value::Byte(_)
            | Value::Short(_)
            | Value::Int(_)
            | Value::Long(_)
            | Value::Float(_)
            | Value::Double(_)
    )
}

/// `value instanceof class` for a non-null value.
pub fn instance_of(v: &Value, class: &str) -> bool {
    if class == DEF || class == "java.lang.Object" {
        return !v.is_null();
    }
    let number = matches!(class, "java.lang.Number");
    let comparable = class == "java.lang.Comparable";
    match v {
        Value::Null => false,
        Value::Bool(_) => matches!(class, "java.lang.Boolean" | "boolean") || comparable,
        Value::Byte(_) => matches!(class, "java.lang.Byte" | "byte") || number || comparable,
        Value::Short(_) => matches!(class, "java.lang.Short" | "short") || number || comparable,
        Value::Char(_) => matches!(class, "java.lang.Character" | "char") || comparable,
        Value::Int(_) => matches!(class, "java.lang.Integer" | "int") || number || comparable,
        Value::Long(_) => matches!(class, "java.lang.Long" | "long") || number || comparable,
        Value::Float(_) => matches!(class, "java.lang.Float" | "float") || number || comparable,
        Value::Double(_) => matches!(class, "java.lang.Double" | "double") || number || comparable,
        Value::Str(_) => {
            matches!(class, "java.lang.String" | "java.lang.CharSequence") || comparable
        }
        Value::List(l) => {
            let kind = l.borrow().kind;
            matches!(
                class,
                "java.util.List" | "java.util.Collection" | "java.lang.Iterable"
            ) || match kind {
                ListKind::ArrayList => {
                    matches!(
                        class,
                        "java.util.ArrayList"
                            | "java.util.AbstractList"
                            | "java.util.AbstractCollection"
                            | "java.util.RandomAccess"
                    )
                }
                ListKind::LinkedList => matches!(
                    class,
                    "java.util.LinkedList"
                        | "java.util.AbstractSequentialList"
                        | "java.util.AbstractList"
                        | "java.util.AbstractCollection"
                        | "java.util.Deque"
                        | "java.util.Queue"
                ),
                ListKind::FixedSize => matches!(
                    class,
                    "java.util.AbstractList"
                        | "java.util.AbstractCollection"
                        | "java.util.RandomAccess"
                ),
                ListKind::Immutable => matches!(class, "java.util.RandomAccess"),
            }
        }
        Value::Map(m) => {
            let kind = m.borrow().kind;
            class == "java.util.Map"
                || match kind {
                    MapKind::HashMap => {
                        matches!(class, "java.util.HashMap" | "java.util.AbstractMap")
                    }
                    MapKind::LinkedHashMap => matches!(
                        class,
                        "java.util.LinkedHashMap" | "java.util.HashMap" | "java.util.AbstractMap"
                    ),
                    MapKind::TreeMap => {
                        matches!(
                            class,
                            "java.util.TreeMap"
                                | "java.util.AbstractMap"
                                | "java.util.SortedMap"
                                | "java.util.NavigableMap"
                        )
                    }
                    MapKind::Immutable => false,
                }
        }
        Value::Set(s) => {
            let kind = s.borrow().kind;
            matches!(
                class,
                "java.util.Set" | "java.util.Collection" | "java.lang.Iterable"
            ) || match kind {
                SetKind::HashSet => matches!(
                    class,
                    "java.util.HashSet" | "java.util.AbstractSet" | "java.util.AbstractCollection"
                ),
                SetKind::LinkedHashSet => {
                    matches!(
                        class,
                        "java.util.LinkedHashSet"
                            | "java.util.HashSet"
                            | "java.util.AbstractSet"
                            | "java.util.AbstractCollection"
                    )
                }
                SetKind::TreeSet => matches!(
                    class,
                    "java.util.TreeSet"
                        | "java.util.AbstractSet"
                        | "java.util.AbstractCollection"
                        | "java.util.SortedSet"
                        | "java.util.NavigableSet"
                ),
                SetKind::Immutable => false,
            }
        }
        Value::Array(a) => {
            let comp = a.borrow().component.clone();
            if let Some(inner) = class.strip_prefix('[') {
                array_assignable(&comp, inner)
            } else {
                matches!(class, "java.lang.Cloneable" | "java.io.Serializable")
            }
        }
        Value::Object(o) => o.instance_of(class),
    }
}

/// Whether an array with component `comp` is an instance of the array type
/// whose descriptor after the leading `[` is `inner`.
fn array_assignable(comp: &str, inner: &str) -> bool {
    let desc = crate::value::descriptor(comp);
    if desc == inner {
        return true;
    }
    match inner.strip_prefix('L').and_then(|s| s.strip_suffix(';')) {
        Some("java.lang.Object") => {
            !matches!(desc.as_str(), "Z" | "B" | "S" | "C" | "I" | "J" | "F" | "D")
        }
        Some(c) => superclasses(comp).contains(&c),
        None => false,
    }
}

/// The classes and interfaces a class (by binary name) is assignable to,
/// for the classes array components and exceptions use.
pub fn superclasses(class: &str) -> Vec<&'static str> {
    let mut v = Vec::new();
    let mut c = class.to_string();
    loop {
        let parent = match c.as_str() {
            "java.lang.Integer" | "java.lang.Long" | "java.lang.Double" | "java.lang.Float"
            | "java.lang.Short" | "java.lang.Byte" => {
                v.push("java.lang.Comparable");
                "java.lang.Number"
            }
            "java.lang.String" => {
                v.push("java.lang.CharSequence");
                v.push("java.lang.Comparable");
                "java.lang.Object"
            }
            "java.util.ArrayList" | "java.util.LinkedList" => {
                v.extend([
                    "java.util.List",
                    "java.util.Collection",
                    "java.lang.Iterable",
                ]);
                "java.util.AbstractList"
            }
            "java.util.HashMap" | "java.util.TreeMap" | "java.util.LinkedHashMap" => {
                v.push("java.util.Map");
                "java.util.AbstractMap"
            }
            "java.util.List" | "java.util.Set" => {
                v.extend(["java.util.Collection", "java.lang.Iterable"]);
                "java.lang.Object"
            }
            other => match exception_parent(other) {
                Some(p) => p,
                None => {
                    if other != "java.lang.Object" {
                        v.push("java.lang.Object");
                    }
                    break;
                }
            },
        };
        v.push(parent);
        c = parent.to_string();
    }
    v
}

/// The superclass of a Java exception class.
pub fn exception_parent(class: &str) -> Option<&'static str> {
    Some(match class {
        "java.lang.Throwable" => "java.lang.Object",
        "java.lang.Exception" | "java.lang.Error" => "java.lang.Throwable",
        "java.lang.RuntimeException"
        | "java.io.IOException"
        | "java.lang.ReflectiveOperationException"
        | "java.lang.CloneNotSupportedException"
        | "java.lang.InterruptedException"
        | "java.text.ParseException"
        | "java.util.TooManyListenersException" => "java.lang.Exception",
        "java.lang.ClassNotFoundException" => "java.lang.ReflectiveOperationException",
        "java.lang.ArithmeticException"
        | "java.lang.ArrayStoreException"
        | "java.lang.ClassCastException"
        | "java.lang.IllegalArgumentException"
        | "java.lang.IllegalStateException"
        | "java.lang.IndexOutOfBoundsException"
        | "java.lang.NegativeArraySizeException"
        | "java.lang.NullPointerException"
        | "java.lang.UnsupportedOperationException"
        | "java.lang.SecurityException"
        | "java.lang.IllegalCallerException"
        | "java.lang.LayerInstantiationException"
        | "java.lang.IllegalMonitorStateException"
        | "java.lang.EnumConstantNotPresentException"
        | "java.lang.TypeNotPresentException"
        | "java.util.NoSuchElementException"
        | "java.util.ConcurrentModificationException"
        | "java.util.EmptyStackException"
        | "java.util.MissingResourceException"
        | "java.time.DateTimeException"
        | "java.io.UncheckedIOException"
        | "org.opensearch.painless.PainlessExplainError" => "java.lang.RuntimeException",
        "java.lang.NumberFormatException"
        | "java.util.regex.PatternSyntaxException"
        | "java.util.IllformedLocaleException"
        | "java.util.IllegalFormatException" => "java.lang.IllegalArgumentException",
        "java.util.UnknownFormatConversionException"
        | "java.util.MissingFormatArgumentException"
        | "java.util.IllegalFormatConversionException"
        | "java.util.DuplicateFormatFlagsException"
        | "java.util.FormatFlagsConversionMismatchException"
        | "java.util.IllegalFormatFlagsException"
        | "java.util.IllegalFormatPrecisionException"
        | "java.util.IllegalFormatWidthException"
        | "java.util.MissingFormatWidthException"
        | "java.util.UnknownFormatFlagsException"
        | "java.util.IllegalFormatCodePointException" => "java.util.IllegalFormatException",
        "java.lang.ArrayIndexOutOfBoundsException"
        | "java.lang.StringIndexOutOfBoundsException" => "java.lang.IndexOutOfBoundsException",
        "java.util.InputMismatchException" => "java.util.NoSuchElementException",
        "java.time.format.DateTimeParseException"
        | "java.time.temporal.UnsupportedTemporalTypeException"
        | "java.time.zone.ZoneRulesException" => "java.time.DateTimeException",
        "java.lang.StackOverflowError" => "java.lang.VirtualMachineError",
        "java.lang.OutOfMemoryError" => "java.lang.VirtualMachineError",
        "java.lang.VirtualMachineError"
        | "java.lang.AssertionError"
        | "org.opensearch.painless.PainlessError"
        | "java.lang.BootstrapMethodError" => "java.lang.Error",
        _ => return None,
    })
}

/// Whether exception class `class` is `target` or a subclass of it.
pub fn exception_is(class: &str, target: &str) -> bool {
    if class == target || target == "java.lang.Object" {
        return true;
    }
    let mut c = class;
    while let Some(p) = exception_parent(c) {
        if p == target {
            return true;
        }
        c = p;
    }
    false
}

/// Painless's canonical type name (`String`, `int[]`, `def`) of a binary
/// name, as error messages print it.
/// `Class.getCanonicalName()` of the class named `name` (its JVM name):
/// `java.util.HashMap.Node`, `int[]`.
pub fn java_canonical(name: &str) -> String {
    if let Some(inner) = name.strip_prefix('[') {
        let base = match inner {
            "Z" => "boolean".to_string(),
            "B" => "byte".into(),
            "S" => "short".into(),
            "C" => "char".into(),
            "I" => "int".into(),
            "J" => "long".into(),
            "F" => "float".into(),
            "D" => "double".into(),
            o if o.starts_with('[') => java_canonical(o),
            o => java_canonical(o.trim_start_matches('L').trim_end_matches(';')),
        };
        return format!("{base}[]");
    }
    name.replace('$', ".")
}

pub fn canonical(name: &str) -> String {
    if name == DEF {
        return "def".into();
    }
    if let Some(inner) = name.strip_prefix('[') {
        let mut dims = 1;
        let mut s = inner;
        while let Some(r) = s.strip_prefix('[') {
            dims += 1;
            s = r;
        }
        let base = match s {
            "Z" => "boolean".to_string(),
            "B" => "byte".into(),
            "S" => "short".into(),
            "C" => "char".into(),
            "I" => "int".into(),
            "J" => "long".into(),
            "F" => "float".into(),
            "D" => "double".into(),
            o => canonical(o.trim_start_matches('L').trim_end_matches(';')),
        };
        return format!("{base}{}", "[]".repeat(dims));
    }
    let simple = name.rsplit('.').next().unwrap_or(name);
    if name.starts_with("java.") || name.starts_with("org.opensearch.") {
        simple.replace('$', ".")
    } else {
        name.replace('$', ".")
    }
}
