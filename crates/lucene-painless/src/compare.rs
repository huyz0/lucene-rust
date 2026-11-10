//! `Comparable.compareTo` for the classes [`Value`] models.

use std::cmp::Ordering;

use crate::value::Value;
use crate::vm::Throw;

/// `String.compareTo`: UTF-16 code units.
pub fn str_cmp(a: &str, b: &str) -> Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// `a.compareTo(b)` when both are `Comparable` with each other.
pub fn natural(a: &Value, b: &Value) -> Option<Ordering> {
    Some(match (a, b) {
        (Value::Int(x), Value::Int(y)) => x.cmp(y),
        (Value::Long(x), Value::Long(y)) => x.cmp(y),
        (Value::Short(x), Value::Short(y)) => x.cmp(y),
        (Value::Byte(x), Value::Byte(y)) => x.cmp(y),
        (Value::Char(x), Value::Char(y)) => x.cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (Value::Double(x), Value::Double(y)) => double_compare(*x, *y),
        (Value::Float(x), Value::Float(y)) => double_compare(f64::from(*x), f64::from(*y)),
        (Value::Str(x), Value::Str(y)) => str_cmp(x, y),
        _ => return None,
    })
}

/// `Double.compare`: `-0.0 < 0.0`, NaN above everything.
pub fn double_compare(x: f64, y: f64) -> Ordering {
    if x < y {
        Ordering::Less
    } else if x > y {
        Ordering::Greater
    } else {
        crate::number::double_to_long_bits(x).cmp(&crate::number::double_to_long_bits(y))
    }
}

/// `a.compareTo(b)`, throwing as Java does for incomparable values.
pub fn compare_to(a: &Value, b: &Value) -> Result<Ordering, Throw> {
    if a.is_null() || b.is_null() {
        return Err(Throw::no_message("java.lang.NullPointerException"));
    }
    natural(a, b).ok_or_else(|| {
        if crate::types::instance_of(a, "java.lang.Comparable") {
            Throw::class_cast(format!(
                "class {} cannot be cast to class {}",
                b.class_name(),
                a.class_name()
            ))
        } else {
            Throw::class_cast(format!(
                "class {} cannot be cast to class java.lang.Comparable",
                a.class_name()
            ))
        }
    })
}

/// A sign as `compareTo` returns it (`Ordering`'s discriminants are Java's
/// `-1`, `0`, `1`).
pub fn ordering_int(o: Ordering) -> i32 {
    o as i32
}
