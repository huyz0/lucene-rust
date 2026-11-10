//! The box classes: `Integer`, `Long`, `Double`, `Float`, `Short`, `Byte`,
//! `Boolean`, `Character` (instance and static methods).

use crate::api::{bool_arg, double_arg, int_arg, long_arg, npe, str_arg};
use crate::compare::{self, ordering_int};
use crate::number::{self, parse_double, parse_long_radix};
use crate::types::{as_f64, as_i64, convert, prim_of, Prim};
use crate::value::{char_string, Value};
use crate::vm::{ApiFn, Throw, Vm};

fn nfe(s: &str) -> Throw {
    Throw::new(
        "java.lang.NumberFormatException",
        format!("For input string: \"{s}\""),
    )
}

/// Methods of a boxed primitive receiver.
pub fn call(
    _vm: &mut Vm,
    recv: &Value,
    name: &str,
    args: &[Value],
) -> Option<Result<Value, Throw>> {
    let p = prim_of(recv)?;
    Some(Ok(match (name, args.len()) {
        ("intValue", 0) if p != Prim::Bool && p != Prim::Char => convert(recv, Prim::Int)?,
        ("longValue", 0) if p != Prim::Bool && p != Prim::Char => convert(recv, Prim::Long)?,
        ("doubleValue", 0) if p != Prim::Bool && p != Prim::Char => convert(recv, Prim::Double)?,
        ("floatValue", 0) if p != Prim::Bool && p != Prim::Char => convert(recv, Prim::Float)?,
        ("shortValue", 0) if p != Prim::Bool && p != Prim::Char => convert(recv, Prim::Short)?,
        ("byteValue", 0) if p != Prim::Bool && p != Prim::Char => convert(recv, Prim::Byte)?,
        ("booleanValue", 0) if p == Prim::Bool => recv.clone(),
        ("charValue", 0) if p == Prim::Char => recv.clone(),
        ("toString", 0) => Value::string(recv.java_to_string()),
        ("hashCode", 0) => Value::Int(recv.java_hash()),
        ("equals", 1) => Value::Bool(recv.java_equals(&args[0])),
        ("compareTo", 1) => {
            // The argument converts to the receiver's type, as the call
            // site adapts a narrower boxed or primitive argument.
            let arg = match prim_of(&args[0]) {
                // `Def`'s implicit conversion to the boxed parameter: a
                // boolean and a char take only their own kind.
                Some(q)
                    if q != p
                        && (matches!(p, Prim::Bool | Prim::Char)
                            || q == Prim::Bool
                            || !crate::def::widens(q, p)) =>
                {
                    return Some(Err(Throw::class_cast(format!(
                        "cannot implicitly cast def [{}] to {}",
                        q.name(),
                        p.boxed()
                    ))));
                }
                Some(q) if q != Prim::Bool && p != Prim::Bool && p != Prim::Char => {
                    if !crate::def::widens(q, p) {
                        return Some(Err(Throw::class_cast(format!(
                            "cannot implicitly cast def [{}] to {}",
                            q.name(),
                            p.boxed()
                        ))));
                    }
                    convert(&args[0], p).unwrap_or_else(|| args[0].clone())
                }
                _ => args[0].clone(),
            };
            match compare::compare_to(recv, &arg) {
                Ok(o) => Value::Int(ordering_int(o)),
                Err(e) => return Some(Err(e)),
            }
        }
        ("isNaN", 0) => Value::Bool(
            as_f64(recv).is_some_and(f64::is_nan) && matches!(p, Prim::Float | Prim::Double),
        ),
        ("isInfinite", 0) => Value::Bool(
            as_f64(recv).is_some_and(f64::is_infinite) && matches!(p, Prim::Float | Prim::Double),
        ),
        _ => return None,
    }))
}

/// `FloatingDecimal.readJavaFormatString(null)`: its `in.trim()`.
fn null_decimal() -> Throw {
    Throw::npe("Cannot invoke \"String.trim()\" because \"in\" is null")
}

fn to_string_radix(v: i64, radix: u32) -> String {
    if !(2..=36).contains(&radix) {
        return v.to_string();
    }
    let neg = v < 0;
    let mut n = v.unsigned_abs();
    if n == 0 {
        return "0".into();
    }
    let mut digits = Vec::new();
    while n > 0 {
        digits.push(std::char::from_digit((n % u64::from(radix)) as u32, radix).unwrap_or('0'));
        n /= u64::from(radix);
    }
    if neg {
        digits.push('-');
    }
    digits.iter().rev().collect()
}

fn parse_int(s: &str, radix: u32, min: i64, max: i64) -> Result<i64, Throw> {
    parse_long_radix(s, radix, min, max).ok_or_else(|| nfe(s))
}

fn char_arg(v: &Value) -> Result<u16, Throw> {
    match v {
        Value::Char(c) => Ok(*c),
        Value::Int(i) => Ok(*i as u16),
        Value::Null => Err(npe()),
        o => Err(Throw::class_cast(format!(
            "{} is not a char",
            o.class_name()
        ))),
    }
}

fn code_point(v: &Value) -> Result<Option<char>, Throw> {
    let cp = match v {
        Value::Char(c) => u32::from(*c),
        other => int_arg(other)? as u32,
    };
    Ok(char::from_u32(cp))
}

pub fn static_lookup(class: &str, name: &str, arity: usize) -> Option<ApiFn> {
    let short = class.rsplit('.').next().unwrap_or(class);
    Some(match (short, name, arity) {
        // Integer
        ("Integer", "parseInt", 1) => |_vm, a| {
            Ok(Value::Int(parse_int(
                &str_arg(&a[0])?,
                10,
                i64::from(i32::MIN),
                i64::from(i32::MAX),
            )? as i32))
        },
        ("Integer", "parseInt", 2) => |_vm, a| {
            Ok(Value::Int(parse_int(
                &str_arg(&a[0])?,
                int_arg(&a[1])? as u32,
                i64::from(i32::MIN),
                i64::from(i32::MAX),
            )? as i32))
        },
        ("Integer", "valueOf", 1) => |_vm, a| match &a[0] {
            Value::Str(s) => {
                Ok(Value::Int(
                    parse_int(s, 10, i64::from(i32::MIN), i64::from(i32::MAX))? as i32,
                ))
            }
            v => Ok(Value::Int(int_arg(v)?)),
        },
        ("Integer", "valueOf", 2) => |_vm, a| {
            Ok(Value::Int(parse_int(
                &str_arg(&a[0])?,
                int_arg(&a[1])? as u32,
                i64::from(i32::MIN),
                i64::from(i32::MAX),
            )? as i32))
        },
        ("Integer", "toString", 1) => |_vm, a| Ok(Value::string(int_arg(&a[0])?.to_string())),
        ("Integer", "toString", 2) => |_vm, a| {
            Ok(Value::string(to_string_radix(
                i64::from(int_arg(&a[0])?),
                int_arg(&a[1])? as u32,
            )))
        },
        ("Integer", "toHexString", 1) => {
            |_vm, a| Ok(Value::string(format!("{:x}", int_arg(&a[0])? as u32)))
        }
        ("Integer", "toBinaryString", 1) => {
            |_vm, a| Ok(Value::string(format!("{:b}", int_arg(&a[0])? as u32)))
        }
        ("Integer", "toOctalString", 1) => {
            |_vm, a| Ok(Value::string(format!("{:o}", int_arg(&a[0])? as u32)))
        }
        ("Integer", "compare", 2) => |_vm, a| {
            Ok(Value::Int(ordering_int(
                int_arg(&a[0])?.cmp(&int_arg(&a[1])?),
            )))
        },
        ("Integer", "max", 2) => |_vm, a| Ok(Value::Int(int_arg(&a[0])?.max(int_arg(&a[1])?))),
        ("Integer", "min", 2) => |_vm, a| Ok(Value::Int(int_arg(&a[0])?.min(int_arg(&a[1])?))),
        ("Integer", "sum", 2) => {
            |_vm, a| Ok(Value::Int(int_arg(&a[0])?.wrapping_add(int_arg(&a[1])?)))
        }
        ("Integer", "signum", 1) => |_vm, a| Ok(Value::Int(int_arg(&a[0])?.signum())),
        ("Integer", "bitCount", 1) => |_vm, a| Ok(Value::Int(int_arg(&a[0])?.count_ones() as i32)),
        ("Integer", "reverse", 1) => |_vm, a| Ok(Value::Int(int_arg(&a[0])?.reverse_bits())),
        ("Integer", "reverseBytes", 1) => |_vm, a| Ok(Value::Int(int_arg(&a[0])?.swap_bytes())),
        ("Integer", "highestOneBit", 1) => |_vm, a| {
            let v = int_arg(&a[0])? as u32;
            Ok(Value::Int(if v == 0 {
                0
            } else {
                (1u32 << (31 - v.leading_zeros())) as i32
            }))
        },
        ("Integer", "lowestOneBit", 1) => |_vm, a| {
            let v = int_arg(&a[0])?;
            Ok(Value::Int(v & v.wrapping_neg()))
        },
        ("Integer", "numberOfLeadingZeros", 1) => {
            |_vm, a| Ok(Value::Int(int_arg(&a[0])?.leading_zeros() as i32))
        }
        ("Integer", "numberOfTrailingZeros", 1) => {
            |_vm, a| Ok(Value::Int(int_arg(&a[0])?.trailing_zeros() as i32))
        }
        ("Integer", "rotateLeft", 2) => |_vm, a| {
            Ok(Value::Int(
                int_arg(&a[0])?.rotate_left((int_arg(&a[1])? & 31) as u32),
            ))
        },
        ("Integer", "rotateRight", 2) => |_vm, a| {
            Ok(Value::Int(
                int_arg(&a[0])?.rotate_right((int_arg(&a[1])? & 31) as u32),
            ))
        },
        ("Integer", "hashCode", 1) => |_vm, a| Ok(Value::Int(int_arg(&a[0])?)),
        ("Integer", "toUnsignedLong", 1) => {
            |_vm, a| Ok(Value::Long(i64::from(int_arg(&a[0])? as u32)))
        }
        ("Integer", "toUnsignedString", 1) => {
            |_vm, a| Ok(Value::string((int_arg(&a[0])? as u32).to_string()))
        }
        ("Integer", "compareUnsigned", 2) => |_vm, a| {
            Ok(Value::Int(ordering_int(
                (int_arg(&a[0])? as u32).cmp(&(int_arg(&a[1])? as u32)),
            )))
        },
        ("Integer", "divideUnsigned", 2) => |_vm, a| {
            let d = int_arg(&a[1])? as u32;
            if d == 0 {
                return Err(Throw::new("java.lang.ArithmeticException", "/ by zero"));
            }
            Ok(Value::Int(((int_arg(&a[0])? as u32) / d) as i32))
        },
        ("Integer", "remainderUnsigned", 2) => |_vm, a| {
            let d = int_arg(&a[1])? as u32;
            if d == 0 {
                return Err(Throw::new("java.lang.ArithmeticException", "/ by zero"));
            }
            Ok(Value::Int(((int_arg(&a[0])? as u32) % d) as i32))
        },
        ("Integer", "parseUnsignedInt", 1) => |_vm, a| {
            let s = str_arg(&a[0])?;
            Ok(Value::Int(
                parse_int(&s, 10, 0, i64::from(u32::MAX))? as u32 as i32,
            ))
        },
        ("Integer", "decode", 1) => |_vm, a| {
            Ok(Value::Int(
                decode(&str_arg(&a[0])?, i64::from(i32::MIN), i64::from(i32::MAX))? as i32,
            ))
        },
        // Long
        ("Long", "parseLong", 1) => |_vm, a| {
            Ok(Value::Long(parse_int(
                &str_arg(&a[0])?,
                10,
                i64::MIN,
                i64::MAX,
            )?))
        },
        ("Long", "parseLong", 2) => |_vm, a| {
            Ok(Value::Long(parse_int(
                &str_arg(&a[0])?,
                int_arg(&a[1])? as u32,
                i64::MIN,
                i64::MAX,
            )?))
        },
        ("Long", "valueOf", 1) => |_vm, a| match &a[0] {
            Value::Str(s) => Ok(Value::Long(parse_int(s, 10, i64::MIN, i64::MAX)?)),
            v => Ok(Value::Long(long_arg(v)?)),
        },
        ("Long", "toString", 1) => |_vm, a| Ok(Value::string(long_arg(&a[0])?.to_string())),
        ("Long", "toString", 2) => |_vm, a| {
            Ok(Value::string(to_string_radix(
                long_arg(&a[0])?,
                int_arg(&a[1])? as u32,
            )))
        },
        ("Long", "toHexString", 1) => {
            |_vm, a| Ok(Value::string(format!("{:x}", long_arg(&a[0])? as u64)))
        }
        ("Long", "toBinaryString", 1) => {
            |_vm, a| Ok(Value::string(format!("{:b}", long_arg(&a[0])? as u64)))
        }
        ("Long", "compare", 2) => |_vm, a| {
            Ok(Value::Int(ordering_int(
                long_arg(&a[0])?.cmp(&long_arg(&a[1])?),
            )))
        },
        ("Long", "max", 2) => |_vm, a| Ok(Value::Long(long_arg(&a[0])?.max(long_arg(&a[1])?))),
        ("Long", "min", 2) => |_vm, a| Ok(Value::Long(long_arg(&a[0])?.min(long_arg(&a[1])?))),
        ("Long", "sum", 2) => {
            |_vm, a| Ok(Value::Long(long_arg(&a[0])?.wrapping_add(long_arg(&a[1])?)))
        }
        ("Long", "signum", 1) => |_vm, a| Ok(Value::Int(long_arg(&a[0])?.signum() as i32)),
        ("Long", "bitCount", 1) => |_vm, a| Ok(Value::Int(long_arg(&a[0])?.count_ones() as i32)),
        ("Long", "numberOfLeadingZeros", 1) => {
            |_vm, a| Ok(Value::Int(long_arg(&a[0])?.leading_zeros() as i32))
        }
        ("Long", "numberOfTrailingZeros", 1) => {
            |_vm, a| Ok(Value::Int(long_arg(&a[0])?.trailing_zeros() as i32))
        }
        ("Long", "hashCode", 1) => {
            |_vm, a| Ok(Value::Int(Value::Long(long_arg(&a[0])?).java_hash()))
        }
        ("Long", "reverse", 1) => |_vm, a| Ok(Value::Long(long_arg(&a[0])?.reverse_bits())),
        // Short / Byte
        ("Short", "parseShort", 1) => |_vm, a| {
            Ok(Value::Short(parse_int(
                &str_arg(&a[0])?,
                10,
                i64::from(i16::MIN),
                i64::from(i16::MAX),
            )? as i16))
        },
        ("Short", "valueOf", 1) => |_vm, a| match &a[0] {
            Value::Str(s) => {
                Ok(Value::Short(
                    parse_int(s, 10, i64::from(i16::MIN), i64::from(i16::MAX))? as i16,
                ))
            }
            v => Ok(convert(v, Prim::Short).ok_or_else(npe)?),
        },
        ("Short", "toString", 1) => |_vm, a| Ok(Value::string(int_arg(&a[0])?.to_string())),
        ("Short", "compare", 2) => |_vm, a| Ok(Value::Int(int_arg(&a[0])? - int_arg(&a[1])?)),
        ("Byte", "parseByte", 1) => |_vm, a| {
            Ok(Value::Byte(
                parse_int(&str_arg(&a[0])?, 10, i64::from(i8::MIN), i64::from(i8::MAX))? as i8,
            ))
        },
        ("Byte", "valueOf", 1) => |_vm, a| match &a[0] {
            Value::Str(s) => {
                Ok(Value::Byte(
                    parse_int(s, 10, i64::from(i8::MIN), i64::from(i8::MAX))? as i8,
                ))
            }
            v => Ok(convert(v, Prim::Byte).ok_or_else(npe)?),
        },
        ("Byte", "toString", 1) => |_vm, a| Ok(Value::string(int_arg(&a[0])?.to_string())),
        ("Byte", "compare", 2) => |_vm, a| Ok(Value::Int(int_arg(&a[0])? - int_arg(&a[1])?)),
        // Double / Float
        ("Double", "parseDouble", 1) | ("Double", "valueOf", 1) => |_vm, a| match &a[0] {
            Value::Str(s) => parse_double(s).map(Value::Double).ok_or_else(|| nfe(s)),
            Value::Null => Err(null_decimal()),
            v => Ok(Value::Double(double_arg(v)?)),
        },
        ("Float", "parseFloat", 1) | ("Float", "valueOf", 1) => |_vm, a| match &a[0] {
            Value::Null => Err(null_decimal()),
            Value::Str(s) => s
                .trim()
                .trim_end_matches(['f', 'F', 'd', 'D'])
                .parse::<f32>()
                .map(Value::Float)
                .map_err(|_| nfe(s)),
            v => Ok(convert(v, Prim::Float).ok_or_else(npe)?),
        },
        ("Double", "toString", 1) => {
            |_vm, a| Ok(Value::string(number::double_to_string(double_arg(&a[0])?)))
        }
        ("Float", "toString", 1) => |_vm, a| {
            Ok(Value::string(number::float_to_string(
                double_arg(&a[0])? as f32
            )))
        },
        ("Double", "compare", 2) | ("Float", "compare", 2) => |_vm, a| {
            Ok(Value::Int(ordering_int(compare::double_compare(
                double_arg(&a[0])?,
                double_arg(&a[1])?,
            ))))
        },
        ("Double", "isNaN", 1) | ("Float", "isNaN", 1) => {
            |_vm, a| Ok(Value::Bool(double_arg(&a[0])?.is_nan()))
        }
        ("Double", "isInfinite", 1) | ("Float", "isInfinite", 1) => {
            |_vm, a| Ok(Value::Bool(double_arg(&a[0])?.is_infinite()))
        }
        ("Double", "isFinite", 1) | ("Float", "isFinite", 1) => {
            |_vm, a| Ok(Value::Bool(double_arg(&a[0])?.is_finite()))
        }
        ("Double", "max", 2) => |_vm, a| {
            Ok(Value::Double(crate::api::math::java_max(
                double_arg(&a[0])?,
                double_arg(&a[1])?,
            )))
        },
        ("Double", "min", 2) => |_vm, a| {
            Ok(Value::Double(crate::api::math::java_min(
                double_arg(&a[0])?,
                double_arg(&a[1])?,
            )))
        },
        ("Double", "sum", 2) => |_vm, a| Ok(Value::Double(double_arg(&a[0])? + double_arg(&a[1])?)),
        ("Double", "doubleToLongBits", 1) => {
            |_vm, a| Ok(Value::Long(number::double_to_long_bits(double_arg(&a[0])?)))
        }
        ("Double", "doubleToRawLongBits", 1) => {
            |_vm, a| Ok(Value::Long(double_arg(&a[0])?.to_bits() as i64))
        }
        ("Double", "longBitsToDouble", 1) => {
            |_vm, a| Ok(Value::Double(f64::from_bits(long_arg(&a[0])? as u64)))
        }
        ("Double", "hashCode", 1) => {
            |_vm, a| Ok(Value::Int(Value::Double(double_arg(&a[0])?).java_hash()))
        }
        ("Float", "floatToIntBits", 1) => |_vm, a| {
            Ok(Value::Int(number::float_to_int_bits(
                double_arg(&a[0])? as f32
            )))
        },
        ("Float", "intBitsToFloat", 1) => {
            |_vm, a| Ok(Value::Float(f32::from_bits(int_arg(&a[0])? as u32)))
        }
        ("Float", "max", 2) => |_vm, a| {
            Ok(Value::Float(
                crate::api::math::java_max(double_arg(&a[0])?, double_arg(&a[1])?) as f32,
            ))
        },
        ("Float", "min", 2) => |_vm, a| {
            Ok(Value::Float(
                crate::api::math::java_min(double_arg(&a[0])?, double_arg(&a[1])?) as f32,
            ))
        },
        // Boolean
        ("Boolean", "parseBoolean", 1) => |_vm, a| {
            Ok(Value::Bool(
                matches!(&a[0], Value::Str(s) if s.eq_ignore_ascii_case("true")),
            ))
        },
        ("Boolean", "valueOf", 1) => |_vm, a| match &a[0] {
            Value::Str(s) => Ok(Value::Bool(s.eq_ignore_ascii_case("true"))),
            Value::Null => Ok(Value::Bool(false)),
            v => Ok(Value::Bool(bool_arg(v)?)),
        },
        ("Boolean", "toString", 1) => |_vm, a| Ok(Value::string(bool_arg(&a[0])?.to_string())),
        ("Boolean", "compare", 2) => |_vm, a| {
            Ok(Value::Int(ordering_int(
                bool_arg(&a[0])?.cmp(&bool_arg(&a[1])?),
            )))
        },
        ("Boolean", "logicalAnd", 2) => {
            |_vm, a| Ok(Value::Bool(bool_arg(&a[0])? && bool_arg(&a[1])?))
        }
        ("Boolean", "logicalOr", 2) => {
            |_vm, a| Ok(Value::Bool(bool_arg(&a[0])? || bool_arg(&a[1])?))
        }
        ("Boolean", "logicalXor", 2) => {
            |_vm, a| Ok(Value::Bool(bool_arg(&a[0])? ^ bool_arg(&a[1])?))
        }
        ("Boolean", "hashCode", 1) => {
            |_vm, a| Ok(Value::Int(if bool_arg(&a[0])? { 1231 } else { 1237 }))
        }
        // Character
        ("Character", "isDigit", 1) => |_vm, a| {
            Ok(Value::Bool(
                code_point(&a[0])?.is_some_and(|c| c.is_ascii_digit()),
            ))
        },
        ("Character", "isLetter", 1) => |_vm, a| {
            Ok(Value::Bool(
                code_point(&a[0])?.is_some_and(char::is_alphabetic),
            ))
        },
        ("Character", "isLetterOrDigit", 1) => |_vm, a| {
            Ok(Value::Bool(
                code_point(&a[0])?.is_some_and(char::is_alphanumeric),
            ))
        },
        ("Character", "isAlphabetic", 1) => |_vm, a| {
            Ok(Value::Bool(
                code_point(&a[0])?.is_some_and(char::is_alphabetic),
            ))
        },
        ("Character", "isWhitespace", 1) => |_vm, a| {
            Ok(Value::Bool(code_point(&a[0])?.is_some_and(|c| {
                c.is_whitespace() && !matches!(c, '\u{00A0}' | '\u{2007}' | '\u{202F}')
                    || matches!(c, '\u{1C}'..='\u{1F}')
            })))
        },
        ("Character", "isSpaceChar", 1) => |_vm, a| {
            Ok(Value::Bool(code_point(&a[0])?.is_some_and(|c| {
                c == ' ' || c == '\u{00A0}' || c == '\u{2007}' || c == '\u{202F}'
            })))
        },
        ("Character", "isUpperCase", 1) => |_vm, a| {
            Ok(Value::Bool(
                code_point(&a[0])?.is_some_and(char::is_uppercase),
            ))
        },
        ("Character", "isLowerCase", 1) => |_vm, a| {
            Ok(Value::Bool(
                code_point(&a[0])?.is_some_and(char::is_lowercase),
            ))
        },
        ("Character", "toUpperCase", 1) => |_vm, a| {
            let c = char_arg(&a[0])?;
            let s = char_string(c).to_uppercase();
            let mut u = s.encode_utf16();
            Ok(Value::Char(match (u.next(), u.next()) {
                (Some(x), None) => x,
                _ => c,
            }))
        },
        ("Character", "toLowerCase", 1) => |_vm, a| {
            let c = char_arg(&a[0])?;
            let s = char_string(c).to_lowercase();
            let mut u = s.encode_utf16();
            Ok(Value::Char(match (u.next(), u.next()) {
                (Some(x), None) => x,
                _ => c,
            }))
        },
        ("Character", "toString", 1) => |_vm, a| Ok(Value::string(char_string(char_arg(&a[0])?))),
        ("Character", "valueOf", 1) => |_vm, a| Ok(Value::Char(char_arg(&a[0])?)),
        ("Character", "compare", 2) => |_vm, a| {
            Ok(Value::Int(
                i32::from(char_arg(&a[0])?) - i32::from(char_arg(&a[1])?),
            ))
        },
        ("Character", "getNumericValue", 1) => |_vm, a| {
            Ok(Value::Int(
                code_point(&a[0])?
                    .and_then(|c| c.to_digit(36))
                    .map_or(-1, |d| d as i32),
            ))
        },
        ("Character", "digit", 2) => |_vm, a| {
            let radix = int_arg(&a[1])? as u32;
            Ok(Value::Int(
                code_point(&a[0])?
                    .and_then(|c| {
                        if (2..=36).contains(&radix) {
                            c.to_digit(radix)
                        } else {
                            None
                        }
                    })
                    .map_or(-1, |d| d as i32),
            ))
        },
        ("Character", "forDigit", 2) => |_vm, a| {
            let (d, r) = (int_arg(&a[0])?, int_arg(&a[1])?);
            Ok(Value::Char(if (2..=36).contains(&r) && d >= 0 && d < r {
                std::char::from_digit(d as u32, r as u32).map_or(0, |c| c as u16)
            } else {
                0
            }))
        },
        ("Character", "hashCode", 1) => |_vm, a| Ok(Value::Int(i32::from(char_arg(&a[0])?))),
        ("Character", "isISOControl", 1) => |_vm, a| {
            Ok(Value::Bool(
                code_point(&a[0])?.is_some_and(char::is_control),
            ))
        },
        ("Character", "charCount", 1) => {
            |_vm, a| Ok(Value::Int(if int_arg(&a[0])? >= 0x10000 { 2 } else { 1 }))
        }
        _ => return None,
    })
}

fn decode(s: &str, min: i64, max: i64) -> Result<i64, Throw> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (radix, digits) = if let Some(r) = body
        .strip_prefix("0x")
        .or_else(|| body.strip_prefix("0X"))
        .or_else(|| body.strip_prefix('#'))
    {
        (16, r)
    } else if body.len() > 1 && body.starts_with('0') {
        (8, &body[1..])
    } else {
        (10, body)
    };
    let text = if neg {
        format!("-{digits}")
    } else {
        digits.to_string()
    };
    parse_long_radix(&text, radix, min, max).ok_or_else(|| nfe(s))
}

#[allow(dead_code)]
fn _i(v: &Value) -> Option<i64> {
    as_i64(v)
}
