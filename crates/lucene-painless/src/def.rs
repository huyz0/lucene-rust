//! Operators with Java's semantics: typed arithmetic (wrapping integers,
//! `ArithmeticException` on integer division by zero, IEEE floats),
//! Painless's `def` operators (`DefMath`: numeric promotion by runtime
//! class, `+` concatenating strings), comparisons and casts.

use crate::types::{
    self, as_f32, as_f64, as_i64, canonical, convert, is_number, prim, unboxed, Prim,
};
use crate::value::Value;
use crate::vm::{BinOp, CastOp, CmpOp, Throw, UnOp, Vm};

const COMPOUND: i32 = 1 << 1;
const EXPLICIT: i32 = 1 << 2;

fn div_zero() -> Throw {
    Throw::new("java.lang.ArithmeticException", "/ by zero")
}

fn null_unbox(v: &Value) -> Result<(), Throw> {
    if v.is_null() {
        Err(Throw::npe("Cannot unbox a null value"))
    } else {
        Ok(())
    }
}

fn int(v: &Value) -> Result<i32, Throw> {
    null_unbox(v)?;
    Ok(as_i64(v).unwrap_or(0) as i32)
}

fn long(v: &Value) -> Result<i64, Throw> {
    null_unbox(v)?;
    Ok(as_i64(v).unwrap_or(0))
}

fn float(v: &Value) -> Result<f32, Throw> {
    null_unbox(v)?;
    Ok(as_f32(v).unwrap_or(0.0))
}

fn double(v: &Value) -> Result<f64, Throw> {
    null_unbox(v)?;
    Ok(as_f64(v).unwrap_or(0.0))
}

fn boolean(v: &Value) -> Result<bool, Throw> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Null => Err(Throw::npe("Cannot unbox a null value")),
        o => Err(Throw::class_cast(format!(
            "Cannot cast [{}] to boolean",
            canonical(&o.class_name())
        ))),
    }
}

/// Arithmetic on two operands already converted to `ty` (`b` is the shift
/// distance for a shift).
pub fn arith(op: BinOp, ty: Prim, a: &Value, b: &Value) -> Result<Value, Throw> {
    Ok(match ty {
        Prim::Int | Prim::Byte | Prim::Short | Prim::Char => {
            let x = int(a)?;
            match op {
                BinOp::Lsh | BinOp::Rsh | BinOp::Ush => {
                    shift_operand(b)?;
                    let s = (long(b)? & 31) as u32;
                    Value::Int(match op {
                        BinOp::Lsh => x.wrapping_shl(s),
                        BinOp::Rsh => x.wrapping_shr(s),
                        _ => ((x as u32) >> s) as i32,
                    })
                }
                _ => {
                    let y = int(b)?;
                    Value::Int(match op {
                        BinOp::Mul => x.wrapping_mul(y),
                        BinOp::Div => {
                            if y == 0 {
                                return Err(div_zero());
                            }
                            x.wrapping_div(y)
                        }
                        BinOp::Rem => {
                            if y == 0 {
                                return Err(div_zero());
                            }
                            x.wrapping_rem(y)
                        }
                        BinOp::Add => x.wrapping_add(y),
                        BinOp::Sub => x.wrapping_sub(y),
                        BinOp::And => x & y,
                        BinOp::Xor => x ^ y,
                        BinOp::Or => x | y,
                        _ => return Err(bad_op(op, ty)),
                    })
                }
            }
        }
        Prim::Long => {
            let x = long(a)?;
            match op {
                BinOp::Lsh | BinOp::Rsh | BinOp::Ush => {
                    shift_operand(b)?;
                    let s = (long(b)? & 63) as u32;
                    Value::Long(match op {
                        BinOp::Lsh => x.wrapping_shl(s),
                        BinOp::Rsh => x.wrapping_shr(s),
                        _ => ((x as u64) >> s) as i64,
                    })
                }
                _ => {
                    let y = long(b)?;
                    Value::Long(match op {
                        BinOp::Mul => x.wrapping_mul(y),
                        BinOp::Div => {
                            if y == 0 {
                                return Err(div_zero());
                            }
                            x.wrapping_div(y)
                        }
                        BinOp::Rem => {
                            if y == 0 {
                                return Err(div_zero());
                            }
                            x.wrapping_rem(y)
                        }
                        BinOp::Add => x.wrapping_add(y),
                        BinOp::Sub => x.wrapping_sub(y),
                        BinOp::And => x & y,
                        BinOp::Xor => x ^ y,
                        BinOp::Or => x | y,
                        _ => return Err(bad_op(op, ty)),
                    })
                }
            }
        }
        Prim::Float => {
            let (x, y) = (float(a)?, float(b)?);
            Value::Float(match op {
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                BinOp::Rem => x % y,
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                _ => return Err(bad_op(op, ty)),
            })
        }
        Prim::Double => {
            let (x, y) = (double(a)?, double(b)?);
            Value::Double(match op {
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                BinOp::Rem => x % y,
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                _ => return Err(bad_op(op, ty)),
            })
        }
        Prim::Bool => {
            let (x, y) = (boolean(a)?, boolean(b)?);
            Value::Bool(match op {
                BinOp::And => x & y,
                BinOp::Xor => x ^ y,
                BinOp::Or => x | y,
                _ => return Err(bad_op(op, ty)),
            })
        }
    })
}

/// A shift distance must be integral (a `def` one reaches here unconverted).
fn shift_operand(b: &Value) -> Result<(), Throw> {
    match b {
        Value::Float(_) | Value::Double(_) => Err(Throw::class_cast(format!(
            "Cannot cast {} to java.lang.Long",
            b.class_name()
        ))),
        _ => Ok(()),
    }
}

fn bad_op(op: BinOp, ty: Prim) -> Throw {
    Throw::class_cast(format!(
        "Cannot apply [{}] operation to type [{}]",
        op.symbol(),
        ty.name()
    ))
}

/// The type `DefMath` promotes two numeric operands to.
fn promote(a: &Value, b: &Value) -> Option<Prim> {
    let num = |v: &Value| is_number(v) || matches!(v, Value::Char(_));
    if !num(a) || !num(b) {
        return None;
    }
    Some(
        if matches!(a, Value::Double(_)) || matches!(b, Value::Double(_)) {
            Prim::Double
        } else if matches!(a, Value::Float(_)) || matches!(b, Value::Float(_)) {
            Prim::Float
        } else if matches!(a, Value::Long(_)) || matches!(b, Value::Long(_)) {
            Prim::Long
        } else {
            Prim::Int
        },
    )
}

/// The type `DefMath` promotes one numeric operand to.
fn promote1(a: &Value) -> Option<Prim> {
    Some(match a {
        Value::Double(_) => Prim::Double,
        Value::Float(_) => Prim::Float,
        Value::Long(_) => Prim::Long,
        Value::Int(_) | Value::Short(_) | Value::Byte(_) | Value::Char(_) => Prim::Int,
        _ => return None,
    })
}

fn class_of(v: &Value) -> String {
    canonical(&v.class_name())
}

/// A `def` binary operator.
pub fn def_arith(
    _vm: &mut Vm,
    op: BinOp,
    a: &Value,
    b: &Value,
    flags: i32,
    ret: Option<&str>,
) -> Result<Value, Throw> {
    if op == BinOp::Add && (matches!(a, Value::Str(_)) || matches!(b, Value::Str(_))) {
        let mut s = a.java_to_string();
        b.write_java(&mut s, 0);
        return Ok(Value::string(s));
    }
    if a.is_null() || b.is_null() {
        return Err(Throw::no_message("java.lang.NullPointerException"));
    }
    let r = match op {
        BinOp::Lsh | BinOp::Rsh | BinOp::Ush => {
            let ty = match a {
                Value::Long(_) => Prim::Long,
                Value::Int(_) | Value::Short(_) | Value::Byte(_) | Value::Char(_) => Prim::Int,
                o => {
                    return Err(Throw::class_cast(format!(
                        "Cannot apply [{}] operation to type [{}].",
                        op.symbol(),
                        class_of(o)
                    )))
                }
            };
            if !matches!(
                b,
                Value::Long(_) | Value::Int(_) | Value::Short(_) | Value::Byte(_) | Value::Char(_)
            ) {
                return Err(Throw::class_cast(format!(
                    "Cannot convert [{}] to an integral value.",
                    class_of(b)
                )));
            }
            arith(op, ty, &convert(a, ty).unwrap_or_default(), b)?
        }
        BinOp::And | BinOp::Or | BinOp::Xor => {
            if let (Value::Bool(_), Value::Bool(_)) = (a, b) {
                arith(op, Prim::Bool, a, b)?
            } else {
                let integral = |v: &Value| {
                    matches!(
                        v,
                        Value::Long(_)
                            | Value::Int(_)
                            | Value::Short(_)
                            | Value::Byte(_)
                            | Value::Char(_)
                    )
                };
                for v in [a, b] {
                    if !integral(v) {
                        return Err(Throw::class_cast(format!(
                            "Cannot convert [{}] to an integral value.",
                            class_of(v)
                        )));
                    }
                }
                let ty = if matches!(a, Value::Long(_)) || matches!(b, Value::Long(_)) {
                    Prim::Long
                } else {
                    Prim::Int
                };
                arith(
                    op,
                    ty,
                    &convert(a, ty).unwrap_or_default(),
                    &convert(b, ty).unwrap_or_default(),
                )?
            }
        }
        _ => {
            let Some(ty) = promote(a, b) else {
                return Err(Throw::class_cast(format!(
                    "Cannot apply [{}] operation to types [{}] and [{}].",
                    op.symbol(),
                    class_of(a),
                    class_of(b)
                )));
            };
            arith(
                op,
                ty,
                &convert(a, ty).unwrap_or_default(),
                &convert(b, ty).unwrap_or_default(),
            )?
        }
    };
    if flags & EXPLICIT != 0 {
        if let Some(ret) = ret {
            return explicit_result(&r, ret);
        }
    } else if flags & COMPOUND != 0 {
        if let Some(p) = types::prim_of(a) {
            return cast_result(&r, p);
        }
    } else if let Some(ret) = ret {
        return call_site_return(r, ret);
    }
    Ok(r)
}

fn cast_result(r: &Value, p: Prim) -> Result<Value, Throw> {
    if (p == Prim::Bool) != matches!(r, Value::Bool(_)) {
        return Err(Throw::class_cast(format!(
            "Cannot cast {} to {}",
            class_of(r),
            p.name()
        )));
    }
    Ok(convert(r, p).unwrap_or_else(|| r.clone()))
}

/// `DefMath.eq`.
pub fn def_eq(a: &Value, b: &Value) -> bool {
    if a.is_null() || b.is_null() {
        return a.is_null() && b.is_null();
    }
    if let Some(ty) = promote(a, b) {
        return num_eq(ty, a, b);
    }
    a.java_equals(b)
}

fn num_eq(ty: Prim, a: &Value, b: &Value) -> bool {
    match ty {
        Prim::Double => as_f64(a) == as_f64(b),
        Prim::Float => as_f32(a) == as_f32(b),
        _ => {
            as_i64(&convert(a, ty).unwrap_or_default())
                == as_i64(&convert(b, ty).unwrap_or_default())
        }
    }
}

/// Reference identity, as `===` sees it.
pub fn identical(a: &Value, b: &Value) -> bool {
    use std::rc::Rc;
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Str(x), Value::Str(y)) => Rc::ptr_eq(x, y) || x == y,
        (Value::List(x), Value::List(y)) => Rc::ptr_eq(x, y),
        (Value::Map(x), Value::Map(y)) => Rc::ptr_eq(x, y),
        (Value::Set(x), Value::Set(y)) => Rc::ptr_eq(x, y),
        (Value::Array(x), Value::Array(y)) => Rc::ptr_eq(x, y),
        (Value::Object(x), Value::Object(y)) => Rc::ptr_eq(x, y),
        _ => a.java_equals(b),
    }
}

/// A comparison. `ty` is the promoted primitive type for a typed numeric
/// comparison; `def` selects `DefMath`; otherwise `==` is `equals`.
pub fn compare(
    op: CmpOp,
    ty: Option<Prim>,
    def: bool,
    a: &Value,
    b: &Value,
) -> Result<bool, Throw> {
    match op {
        CmpOp::EqR => return Ok(identical(a, b)),
        CmpOp::NeR => return Ok(!identical(a, b)),
        CmpOp::Eq | CmpOp::Ne => {
            let eq = if let Some(p) = ty {
                null_unbox(a)?;
                null_unbox(b)?;
                match p {
                    Prim::Bool => boolean(a)? == boolean(b)?,
                    _ => num_eq(p, a, b),
                }
            } else if def {
                def_eq(a, b)
            } else {
                match (a, b) {
                    (Value::Null, Value::Null) => true,
                    (Value::Null, _) | (_, Value::Null) => false,
                    _ => a.java_equals(b),
                }
            };
            return Ok(eq == (op == CmpOp::Eq));
        }
        _ => {}
    }
    let p = match ty {
        Some(p) => {
            null_unbox(a)?;
            null_unbox(b)?;
            p
        }
        None => {
            if a.is_null() || b.is_null() {
                return Err(Throw::no_message("java.lang.NullPointerException"));
            }
            match promote(a, b) {
                Some(p) => p,
                None => {
                    let sym = match op {
                        CmpOp::Lt => "<",
                        CmpOp::Lte => "<=",
                        CmpOp::Gt => ">",
                        _ => ">=",
                    };
                    return Err(Throw::class_cast(format!(
                        "Cannot apply [{sym}] operation to types [{}] and [{}].",
                        class_of(a),
                        class_of(b)
                    )));
                }
            }
        }
    };
    Ok(match p {
        Prim::Float | Prim::Double => {
            let (x, y) = if p == Prim::Float {
                (f64::from(float(a)?), f64::from(float(b)?))
            } else {
                (double(a)?, double(b)?)
            };
            match op {
                CmpOp::Lt => x < y,
                CmpOp::Lte => x <= y,
                CmpOp::Gt => x > y,
                _ => x >= y,
            }
        }
        _ => {
            let (x, y) = (long(a)?, long(b)?);
            match op {
                CmpOp::Lt => x < y,
                CmpOp::Lte => x <= y,
                CmpOp::Gt => x > y,
                _ => x >= y,
            }
        }
    })
}

/// A unary operator: typed (`ty`) or `def`.
pub fn unary(
    op: UnOp,
    ty: Option<Prim>,
    v: &Value,
    ret: Option<&str>,
    explicit: bool,
) -> Result<Value, Throw> {
    if op == UnOp::Not {
        return Ok(Value::Bool(!boolean(v)?));
    }
    let p = match ty {
        Some(p) => {
            null_unbox(v)?;
            p
        }
        None => {
            if v.is_null() {
                return Err(Throw::no_message("java.lang.NullPointerException"));
            }
            let p = promote1(v);
            match (op, p) {
                (UnOp::BwNot, Some(Prim::Long | Prim::Int)) | (UnOp::Neg | UnOp::Plus, Some(_)) => {
                    p.unwrap_or(Prim::Int)
                }
                _ => {
                    let sym = match op {
                        UnOp::Neg => "-",
                        UnOp::Plus => "+",
                        _ => "~",
                    };
                    return Err(Throw::class_cast(format!(
                        "Cannot apply [{sym}] operation to type [{}].",
                        class_of(v)
                    )));
                }
            }
        }
    };
    let x = convert(v, p).unwrap_or_default();
    let r = match (op, x) {
        (UnOp::Plus, x) => x,
        (UnOp::Neg, Value::Int(i)) => Value::Int(i.wrapping_neg()),
        (UnOp::Neg, Value::Long(i)) => Value::Long(i.wrapping_neg()),
        (UnOp::Neg, Value::Float(f)) => Value::Float(-f),
        (UnOp::Neg, Value::Double(f)) => Value::Double(-f),
        (UnOp::BwNot, Value::Int(i)) => Value::Int(!i),
        (UnOp::BwNot, Value::Long(i)) => Value::Long(!i),
        (_, x) => {
            return Err(Throw::class_cast(format!(
                "Cannot apply operation to type [{}].",
                class_of(&x)
            )))
        }
    };
    if ty.is_none() {
        if let Some(ret) = ret {
            return if explicit {
                explicit_result(&r, ret)
            } else {
                call_site_return(r, ret)
            };
        }
    }
    Ok(r)
}

fn explicit_result(r: &Value, ret: &str) -> Result<Value, Throw> {
    match prim(ret).or_else(|| unboxed(ret)) {
        Some(p) => cast_result(r, p),
        None => Ok(r.clone()),
    }
}

/// Primitive widening, as `MethodHandle.asType` adapts a `def` operator's
/// result to its call site's type.
pub fn widens(from: Prim, to: Prim) -> bool {
    use Prim::*;
    from == to
        || matches!(
            (from, to),
            (Byte, Short | Int | Long | Float | Double)
                | (Short, Int | Long | Float | Double)
                | (Char, Int | Long | Float | Double)
                | (Int, Long | Float | Double)
                | (Long, Float | Double)
                | (Float, Double)
        )
}

/// A `def` result converted to the call site's type `ret`.
pub fn call_site_return(r: Value, ret: &str) -> Result<Value, Throw> {
    if let Some(p) = prim(ret) {
        let Some(from) = types::prim_of(&r) else {
            return if r.is_null() {
                Err(Throw::no_message("java.lang.NullPointerException"))
            } else {
                Ok(r)
            };
        };
        if !widens(from, p) {
            return Err(Throw::class_cast(format!(
                "Cannot cast from: {} to {}",
                from.name(),
                p.name()
            )));
        }
        return Ok(convert(&r, p).unwrap_or(r));
    }
    if unboxed(ret).is_some() && !r.is_null() && !types::instance_of(&r, ret) {
        return Err(Throw::class_cast(format!(
            "Cannot cast {} to {}",
            r.class_name(),
            ret
        )));
    }
    Ok(r)
}

fn unboxed_name(v: &Value) -> String {
    match types::prim_of(v) {
        Some(p) => p.name().to_string(),
        None => canonical(&v.class_name()),
    }
}

/// A `PainlessCast`.
pub fn cast(v: &Value, c: &CastOp) -> Result<Value, Throw> {
    match c {
        CastOp::CharToString => match v {
            Value::Char(ch) => Ok(Value::string(crate::value::char_string(*ch))),
            Value::Null => Err(Throw::no_message("java.lang.NullPointerException")),
            o => Ok(o.clone()),
        },
        CastOp::StringToChar => match v {
            Value::Str(s) => {
                let mut u = s.encode_utf16();
                match (u.next(), u.next()) {
                    (Some(c), None) => Ok(Value::Char(c)),
                    _ => Err(Throw::class_cast(format!(
                        "cannot cast java.lang.String with length not equal to one to char"
                    ))),
                }
            }
            Value::Null => Err(Throw::class_cast(
                "cannot cast null java.lang.String to char",
            )),
            o => Ok(o.clone()),
        },
        CastOp::Prim { to, unbox } => {
            if v.is_null() {
                return if *unbox {
                    Err(Throw::no_message("java.lang.NullPointerException"))
                } else {
                    Ok(Value::Null)
                };
            }
            Ok(convert(v, *to).unwrap_or_else(|| v.clone()))
        }
        CastOp::DefTo {
            to,
            explicit,
            boxed,
        } => {
            if v.is_null() {
                return if *boxed {
                    Ok(Value::Null)
                } else {
                    Err(Throw::no_message("java.lang.NullPointerException"))
                };
            }
            if let (Prim::Char, true, Value::Str(s)) = (to, explicit, v) {
                return cast(v, &CastOp::StringToChar).map_err(|_| {
                    Throw::class_cast(format!(
                        "cannot cast java.lang.String with length not equal to one to char: {s}"
                    ))
                });
            }
            let ok = match (to, explicit) {
                (Prim::Bool, _) => matches!(v, Value::Bool(_)),
                (_, true) => is_number(v) || matches!(v, Value::Char(_)),
                (Prim::Byte, false) => matches!(v, Value::Byte(_)),
                (Prim::Short, false) => matches!(v, Value::Byte(_) | Value::Short(_)),
                (Prim::Char, false) => matches!(v, Value::Char(_)),
                (Prim::Int, false) => matches!(
                    v,
                    Value::Byte(_) | Value::Short(_) | Value::Char(_) | Value::Int(_)
                ),
                (Prim::Long, false) => matches!(
                    v,
                    Value::Byte(_)
                        | Value::Short(_)
                        | Value::Char(_)
                        | Value::Int(_)
                        | Value::Long(_)
                ),
                (Prim::Float, false) => {
                    !matches!(v, Value::Double(_)) && (is_number(v) || matches!(v, Value::Char(_)))
                }
                (Prim::Double, false) => is_number(v) || matches!(v, Value::Char(_)),
            };
            if !ok {
                let how = if *to == Prim::Bool {
                    "cannot cast"
                } else if *explicit {
                    "cannot explicitly cast"
                } else {
                    "cannot implicitly cast"
                };
                let target = if *boxed {
                    canonical(to.boxed())
                } else {
                    to.name().to_string()
                };
                return Err(Throw::class_cast(format!(
                    "{how} def [{}] to {target}",
                    unboxed_name(v)
                )));
            }
            Ok(convert(v, *to).unwrap_or_else(|| v.clone()))
        }
        CastOp::DefToString { explicit } => match v {
            Value::Null | Value::Str(_) => Ok(v.clone()),
            Value::Char(c) if *explicit => Ok(Value::string(crate::value::char_string(*c))),
            o => Err(Throw::class_cast(format!(
                "cannot {} cast def [{}] to java.lang.String",
                if *explicit {
                    "explicitly"
                } else {
                    "implicitly"
                },
                unboxed_name(o)
            ))),
        },
        CastOp::Check(class) => {
            if v.is_null() || types::instance_of(v, class) {
                Ok(v.clone())
            } else {
                Err(Throw::class_cast(format!(
                    "class {} cannot be cast to class {}",
                    v.class_name(),
                    class
                )))
            }
        }
    }
}
