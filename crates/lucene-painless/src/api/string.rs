//! `String`, `CharSequence` and `StringBuilder`, indexed in UTF-16 units
//! as Java's.

use std::cell::RefCell;
use std::rc::Rc;

use crate::api::{self, bool_arg, int_arg, npe, str_arg};
use crate::compare;
use crate::object::JObject;
use crate::value::{char_string, string_hash, Value};
use crate::vm::{ApiFn, Throw, Vm};

pub fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

pub fn from_utf16(u: &[u16]) -> String {
    String::from_utf16_lossy(u)
}

fn sioobe(msg: String) -> Throw {
    Throw::new("java.lang.StringIndexOutOfBoundsException", msg)
}

/// `s.substring(a, b)` in UTF-16 units.
pub fn substring(s: &str, a: i32, b: i32) -> Result<String, Throw> {
    let len = if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    } as i32;
    if a < 0 || b > len || a > b {
        return Err(sioobe(format!("begin {a}, end {b}, length {len}")));
    }
    if s.is_ascii() {
        return Ok(s[a as usize..b as usize].to_string());
    }
    let u = utf16(s);
    Ok(from_utf16(&u[a as usize..b as usize]))
}

fn len16(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}

fn char_at(s: &str, i: i32) -> Result<u16, Throw> {
    let u: Vec<u16> = if s.is_ascii() {
        s.bytes().map(u16::from).collect()
    } else {
        utf16(s)
    };
    usize::try_from(i)
        .ok()
        .and_then(|i| u.get(i).copied())
        .ok_or_else(|| sioobe(format!("Index {i} out of bounds for length {}", u.len())))
}

/// `indexOf` in UTF-16 units from `from`.
fn index_of(s: &str, needle: &str, from: i32) -> i32 {
    let (h, n) = (utf16(s), utf16(needle));
    let from = from.max(0) as usize;
    if n.is_empty() {
        return from.min(h.len()) as i32;
    }
    if n.len() > h.len() {
        return -1;
    }
    (from..=h.len() - n.len())
        .find(|&i| h[i..i + n.len()] == n[..])
        .map_or(-1, |i| i as i32)
}

fn last_index_of(s: &str, needle: &str, from: i32) -> i32 {
    let (h, n) = (utf16(s), utf16(needle));
    if n.len() > h.len() || from < 0 {
        return -1;
    }
    let start = (from as usize).min(h.len() - n.len());
    (0..=start)
        .rev()
        .find(|&i| h[i..i + n.len()] == n[..])
        .map_or(-1, |i| i as i32)
}

fn java_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c <= ' ')
}

fn lower(s: &str) -> String {
    s.to_lowercase()
}

/// Methods of a `String` receiver.
pub fn call(vm: &mut Vm, s: &Rc<str>, name: &str, args: &[Value]) -> Option<Result<Value, Throw>> {
    let r = (|| -> Result<Value, Throw> {
        Ok(match (name, args.len()) {
            ("length", 0) => Value::Int(len16(s) as i32),
            ("isEmpty", 0) => Value::Bool(s.is_empty()),
            ("charAt", 1) => Value::Char(char_at(s, int_arg(&args[0])?)?),
            ("substring", 1) => {
                let a = int_arg(&args[0])?;
                Value::string(substring(s, a, len16(s) as i32)?)
            }
            ("substring", 2) | ("subSequence", 2) => {
                Value::string(substring(s, int_arg(&args[0])?, int_arg(&args[1])?)?)
            }
            ("indexOf", 1) => Value::Int(index_of(s, &str_arg(&args[0])?, 0)),
            ("indexOf", 2) => Value::Int(index_of(s, &str_arg(&args[0])?, int_arg(&args[1])?)),
            ("lastIndexOf", 1) => Value::Int(last_index_of(s, &str_arg(&args[0])?, i32::MAX)),
            ("lastIndexOf", 2) => {
                Value::Int(last_index_of(s, &str_arg(&args[0])?, int_arg(&args[1])?))
            }
            ("contains", 1) => Value::Bool(s.contains(&*str_arg(&args[0])?)),
            ("startsWith", 1) => Value::Bool(s.starts_with(&*str_arg(&args[0])?)),
            ("startsWith", 2) => {
                let off = int_arg(&args[1])?;
                let p = str_arg(&args[0])?;
                let u = utf16(s);
                let pu = utf16(&p);
                Value::Bool(
                    off >= 0
                        && (off as usize) + pu.len() <= u.len()
                        && u[off as usize..off as usize + pu.len()] == pu[..],
                )
            }
            ("endsWith", 1) => Value::Bool(s.ends_with(&*str_arg(&args[0])?)),
            ("equals", 1) => Value::Bool(matches!(&args[0], Value::Str(o) if **o == **s)),
            ("equalsIgnoreCase", 1) => Value::Bool(match &args[0] {
                Value::Str(o) => {
                    let (a, b) = (utf16(s), utf16(o));
                    a.len() == b.len()
                        && a.iter().zip(&b).all(|(x, y)| {
                            x == y || {
                                let (cx, cy) = (char_string(*x), char_string(*y));
                                cx.to_uppercase() == cy.to_uppercase()
                                    || cx.to_lowercase() == cy.to_lowercase()
                            }
                        })
                }
                _ => false,
            }),
            ("hashCode", 0) => Value::Int(string_hash(s)),
            ("toString", 0) => Value::Str(s.clone()),
            ("compareTo", 1) => {
                let o = str_arg(&args[0])?;
                Value::Int(java_compare(s, &o))
            }
            ("compareToIgnoreCase", 1) => {
                let o = str_arg(&args[0])?;
                Value::Int(java_compare(
                    &lower(&s.to_uppercase()),
                    &lower(&o.to_uppercase()),
                ))
            }
            ("concat", 1) => Value::string(format!("{s}{}", str_arg(&args[0])?)),
            ("contentEquals", 1) => Value::Bool(args[0].java_to_string() == **s),
            ("replace", 2) => match (&args[0], &args[1]) {
                (Value::Char(a), Value::Char(b)) => {
                    Value::string(s.replace(&char_string(*a), &char_string(*b)))
                }
                (a, b) => {
                    let (a, b) = (str_arg(a)?, str_arg(b)?);
                    if a.is_empty() {
                        let mut out = String::new();
                        out.push_str(&b);
                        for c in s.chars() {
                            out.push(c);
                            out.push_str(&b);
                        }
                        Value::string(out)
                    } else {
                        Value::string(s.replace(&*a, &b))
                    }
                }
            },
            ("toLowerCase", 0) | ("toLowerCase", 1) => Value::string(lower(s)),
            ("toUpperCase", 0) | ("toUpperCase", 1) => Value::string(s.to_uppercase()),
            ("trim", 0) => Value::str(java_trim(s)),
            ("strip", 0) => Value::str(s.trim()),
            ("toCharArray", 0) => {
                Value::array("char", utf16(s).into_iter().map(Value::Char).collect())
            }
            ("chars", 0) => return Err(crate::compile::missing("String.chars")),
            ("codePointAt", 1) => {
                let i = int_arg(&args[0])?;
                let u = utf16(s);
                let c = *u
                    .get(usize::try_from(i).unwrap_or(usize::MAX))
                    .ok_or_else(|| {
                        sioobe(format!("Index {i} out of bounds for length {}", u.len()))
                    })?;
                let cp = if (0xD800..0xDC00).contains(&c) {
                    match u.get(i as usize + 1) {
                        Some(&d) if (0xDC00..0xE000).contains(&d) => {
                            0x10000 + ((u32::from(c) - 0xD800) << 10) + (u32::from(d) - 0xDC00)
                        }
                        _ => u32::from(c),
                    }
                } else {
                    u32::from(c)
                };
                Value::Int(cp as i32)
            }
            ("codePointCount", 2) => {
                let (a, b) = (int_arg(&args[0])?, int_arg(&args[1])?);
                Value::Int(substring(s, a, b)?.chars().count() as i32)
            }
            ("splitOnToken", 1) | ("splitOnToken", 2) => {
                let token = str_arg(&args[0])?;
                let limit = if args.len() == 2 {
                    int_arg(&args[1])?
                } else {
                    0
                };
                Value::array(
                    "java.lang.String",
                    split_on_token(s, &token, limit)?
                        .into_iter()
                        .map(Value::string)
                        .collect(),
                )
            }
            ("encodeBase64", 0) => Value::string(base64_encode(s.as_bytes())),
            ("decodeBase64", 0) => {
                let b = base64_decode(s)
                    .ok_or_else(|| Throw::illegal_arg("Illegal base64 character"))?;
                Value::string(String::from_utf8_lossy(&b).into_owned())
            }
            ("regionMatches", 4) | ("regionMatches", 5) => {
                let (ic, o) = if args.len() == 5 {
                    (bool_arg(&args[0])?, 1)
                } else {
                    (false, 0)
                };
                let to = int_arg(&args[o])?;
                let other = str_arg(&args[o + 1])?;
                let oo = int_arg(&args[o + 2])?;
                let len = int_arg(&args[o + 3])?;
                let (a, b) = (utf16(s), utf16(&other));
                let ok = to >= 0
                    && oo >= 0
                    && (to as i64 + len as i64) <= a.len() as i64
                    && (oo as i64 + len as i64) <= b.len() as i64
                    && (0..len.max(0) as usize).all(|i| {
                        let (x, y) = (a[to as usize + i], b[oo as usize + i]);
                        x == y
                            || ic && char_string(x).to_uppercase() == char_string(y).to_uppercase()
                    });
                Value::Bool(ok)
            }
            ("replaceAll", 2) | ("replaceFirst", 2) => {
                return api::regex::replace_with(
                    vm,
                    &Value::Str(s.clone()),
                    &args[0],
                    &args[1],
                    name == "replaceAll",
                )
            }
            ("matches", 1) => return Err(crate::compile::missing("String.matches")),
            _ => return Err(Throw::new(api::UNSUPPORTED, "")),
        })
    })();
    match r {
        Err(t) if t.class() == api::UNSUPPORTED && t.message().as_deref() == Some("") => None,
        r => Some(r),
    }
}

/// `String.compareTo`: the first differing UTF-16 unit, else the length.
pub fn java_compare(a: &str, b: &str) -> i32 {
    let (x, y) = (utf16(a), utf16(b));
    for (p, q) in x.iter().zip(&y) {
        if p != q {
            return i32::from(*p) - i32::from(*q);
        }
    }
    x.len() as i32 - y.len() as i32
}

fn split_on_token(s: &str, token: &str, limit: i32) -> Result<Vec<String>, Throw> {
    if token.is_empty() {
        return Err(Throw::illegal_arg("empty token"));
    }
    let mut out: Vec<String> = Vec::new();
    let mut rest = s;
    loop {
        if limit > 0 && out.len() as i32 == limit - 1 {
            out.push(rest.to_string());
            break;
        }
        match rest.find(token) {
            Some(i) => {
                out.push(rest[..i].to_string());
                rest = &rest[i + token.len()..];
            }
            None => {
                out.push(rest.to_string());
                break;
            }
        }
    }
    if limit == 0 {
        while out.last().is_some_and(String::is_empty) {
            out.pop();
        }
    }
    let _ = compare::str_cmp;
    Ok(out)
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                out.push(B64[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        if c == b'=' {
            break;
        }
        let v = B64.iter().position(|&x| x == c)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// `String.valueOf` and friends.
pub fn static_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("valueOf", 1) => |_vm, a| {
            Ok(Value::string(match &a[0] {
                Value::Array(arr) if &*arr.borrow().component == "char" => arr
                    .borrow()
                    .items
                    .iter()
                    .map(|c| if let Value::Char(c) = c { *c } else { 0 })
                    .map(|c| char_string(c))
                    .collect(),
                v => v.java_to_string(),
            }))
        },
        ("copyValueOf", 1) => |_vm, a| Ok(Value::string(char_array_string(&a[0])?)),
        ("join", 2) => |vm, a| {
            let sep = a[0].java_to_string();
            let items = match &a[1] {
                Value::Array(arr) => arr.borrow().items.clone(),
                v => api::collection::elements(vm, v)?,
            };
            Ok(Value::string(
                items
                    .iter()
                    .map(Value::java_to_string)
                    .collect::<Vec<_>>()
                    .join(&sep),
            ))
        },
        ("format", 2) => |_vm, a| Ok(Value::string(format(&str_arg(&a[0])?, &varargs(&a[1]))?)),
        ("format", 3) => |_vm, a| Ok(Value::string(format(&str_arg(&a[1])?, &varargs(&a[2]))?)),
        _ => return None,
    })
}

fn varargs(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(a) => a.borrow().items.clone(),
        other => vec![other.clone()],
    }
}

fn char_array_string(v: &Value) -> Result<String, Throw> {
    match v {
        Value::Array(a) => Ok(from_utf16(
            &a.borrow()
                .items
                .iter()
                .map(|c| if let Value::Char(c) = c { *c } else { 0 })
                .collect::<Vec<_>>(),
        )),
        Value::Null => Err(npe()),
        _ => Err(Throw::class_cast("not a char[]")),
    }
}

/// `java.util.Formatter` for the common conversions.
pub fn format(fmt: &str, args: &[Value]) -> Result<String, Throw> {
    let mut out = String::new();
    let mut chars = fmt.chars().peekable();
    let mut next_arg = 0usize;
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let mut spec = String::new();
        let conv = loop {
            match chars.next() {
                Some(ch)
                    if "-#+ 0,(".contains(ch) || ch.is_ascii_digit() || ch == '.' || ch == '$' =>
                {
                    spec.push(ch)
                }
                Some(ch) => break ch,
                None => {
                    return Err(Throw::new(
                        "java.util.UnknownFormatConversionException",
                        "Conversion = '%'",
                    ))
                }
            }
        };
        match conv {
            '%' => {
                out.push('%');
                continue;
            }
            'n' => {
                out.push('\n');
                continue;
            }
            _ => {}
        }
        let (index, spec) = match spec.find('$') {
            Some(p) => (
                spec[..p].parse::<usize>().ok().map(|i| i - 1),
                spec[p + 1..].to_string(),
            ),
            None => (None, spec),
        };
        let arg = match index {
            Some(i) => args.get(i).cloned(),
            None => {
                let a = args.get(next_arg).cloned();
                next_arg += 1;
                a
            }
        }
        .ok_or_else(|| {
            Throw::new(
                "java.util.MissingFormatArgumentException",
                format!("Format specifier '%{spec}{conv}'"),
            )
        })?;
        let left = spec.contains('-');
        let zero = spec.starts_with('0') || spec.contains(",0");
        let plus = spec.contains('+');
        let comma = spec.contains(',');
        let digits: String = spec
            .trim_start_matches(['-', '+', ' ', '#', ',', '('])
            .to_string();
        let (width, prec) = match digits.split_once('.') {
            Some((w, p)) => (
                w.trim_start_matches('0').parse::<usize>().ok(),
                p.parse::<usize>().ok(),
            ),
            None => (digits.trim_start_matches('0').parse::<usize>().ok(), None),
        };
        let body = match conv {
            's' | 'S' => {
                let s = arg.java_to_string();
                let s = match prec {
                    Some(p) => s.chars().take(p).collect(),
                    None => s,
                };
                if conv == 'S' {
                    s.to_uppercase()
                } else {
                    s
                }
            }
            'b' | 'B' => match &arg {
                Value::Null => "false".into(),
                Value::Bool(b) => b.to_string(),
                _ => "true".into(),
            },
            'c' => match &arg {
                Value::Char(c) => char_string(*c),
                Value::Int(i) => char::from_u32(*i as u32)
                    .map(String::from)
                    .unwrap_or_default(),
                o => return Err(illegal_conv(conv, o)),
            },
            'd' => match &arg {
                Value::Int(_) | Value::Long(_) | Value::Short(_) | Value::Byte(_) => {
                    let v = crate::types::as_i64(&arg).unwrap_or(0);
                    let mut s = v.unsigned_abs().to_string();
                    if comma {
                        s = group(&s);
                    }
                    if v < 0 {
                        format!("-{s}")
                    } else if plus {
                        format!("+{s}")
                    } else {
                        s
                    }
                }
                o => return Err(illegal_conv(conv, o)),
            },
            'x' | 'X' => {
                let s = match &arg {
                    Value::Int(i) => format!("{:x}", *i as u32),
                    Value::Long(i) => format!("{:x}", *i as u64),
                    o => return Err(illegal_conv(conv, o)),
                };
                if conv == 'X' {
                    s.to_uppercase()
                } else {
                    s
                }
            }
            'o' => match &arg {
                Value::Int(i) => format!("{:o}", *i as u32),
                Value::Long(i) => format!("{:o}", *i as u64),
                o => return Err(illegal_conv(conv, o)),
            },
            'f' | 'e' | 'E' | 'g' => match &arg {
                Value::Float(_) | Value::Double(_) => {
                    let v = crate::types::as_f64(&arg).unwrap_or(0.0);
                    let p = prec.unwrap_or(6);
                    let mut s = match conv {
                        'f' => format_fixed(v, p),
                        'e' | 'E' => {
                            let s = format!("{:.*e}", p, v);
                            let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
                            let e: i32 = e.parse().unwrap_or(0);
                            let s = format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
                            if conv == 'E' {
                                s.to_uppercase()
                            } else {
                                s
                            }
                        }
                        _ => format!("{v}"),
                    };
                    if comma {
                        let (i, f) = s.split_once('.').map_or((s.clone(), None), |(i, f)| {
                            (i.to_string(), Some(f.to_string()))
                        });
                        let neg = i.starts_with('-');
                        let g = group(i.trim_start_matches('-'));
                        s = format!(
                            "{}{g}{}",
                            if neg { "-" } else { "" },
                            f.map(|f| format!(".{f}")).unwrap_or_default()
                        );
                    }
                    if plus && v >= 0.0 {
                        s = format!("+{s}");
                    }
                    s
                }
                o => return Err(illegal_conv(conv, o)),
            },
            other => {
                return Err(Throw::new(
                    "java.util.UnknownFormatConversionException",
                    format!("Conversion = '{other}'"),
                ))
            }
        };
        let pad = width.unwrap_or(0).saturating_sub(body.chars().count());
        if left {
            out.push_str(&body);
            out.extend(std::iter::repeat_n(' ', pad));
        } else if zero && conv != 's' {
            let (sign, digits) = if body.starts_with(['-', '+']) {
                body.split_at(1)
            } else {
                ("", body.as_str())
            };
            out.push_str(sign);
            out.extend(std::iter::repeat_n('0', pad));
            out.push_str(digits);
        } else {
            out.extend(std::iter::repeat_n(' ', pad));
            out.push_str(&body);
        }
    }
    Ok(out)
}

fn illegal_conv(conv: char, v: &Value) -> Throw {
    Throw::new(
        "java.util.IllegalFormatConversionException",
        format!("{conv} != {}", v.class_name()),
    )
}

fn group(digits: &str) -> String {
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `%.Nf` with Java's HALF_UP rounding of the exact decimal value.
pub fn format_fixed(v: f64, p: usize) -> String {
    if !v.is_finite() {
        return if v.is_nan() {
            "NaN".into()
        } else if v > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    // Java rounds the shortest repr's decimal digits HALF_UP.
    let s = crate::number::double_to_string(v.abs());
    let plain = to_plain(&s);
    let (i, f) = plain.split_once('.').unwrap_or((&plain, ""));
    let mut digits: Vec<u8> = i.bytes().chain(f.bytes()).map(|b| b - b'0').collect();
    let int_len = i.len();
    let keep = int_len + p;
    let mut round_up = digits.get(keep).is_some_and(|&d| d >= 5);
    digits.truncate(keep);
    while digits.len() < keep {
        digits.push(0);
    }
    let mut k = digits.len();
    while round_up && k > 0 {
        k -= 1;
        if digits[k] == 9 {
            digits[k] = 0;
        } else {
            digits[k] += 1;
            round_up = false;
        }
    }
    let mut int_part: String = digits[..int_len]
        .iter()
        .map(|d| (d + b'0') as char)
        .collect();
    if round_up {
        int_part.insert(0, '1');
    }
    let int_part = int_part.trim_start_matches('0');
    let int_part = if int_part.is_empty() { "0" } else { int_part };
    let frac: String = digits[int_len..]
        .iter()
        .map(|d| (d + b'0') as char)
        .collect();
    let neg = v.is_sign_negative() && (int_part != "0" || frac.bytes().any(|b| b != b'0'));
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    out.push_str(int_part);
    if p > 0 {
        out.push('.');
        out.push_str(&frac);
    }
    out
}

/// `1.5E-4` → `0.00015`.
fn to_plain(s: &str) -> String {
    let Some((m, e)) = s.split_once('E') else {
        return s.to_string();
    };
    let e: i32 = e.parse().unwrap_or(0);
    let (i, f) = m.split_once('.').unwrap_or((m, ""));
    let digits = format!("{i}{f}");
    let point = i.len() as i32 + e;
    if point <= 0 {
        format!("0.{}{}", "0".repeat((-point) as usize), digits)
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    }
}

pub fn string_constructor(arity: usize) -> Option<ApiFn> {
    Some(match arity {
        0 => |_vm, _a| Ok(Value::str("")),
        _ => return None,
    })
}

/// A `StringBuilder`.
#[derive(Debug, Default)]
pub struct JStringBuilder {
    pub buf: RefCell<String>,
}

pub fn builder_constructor(arity: usize) -> Option<ApiFn> {
    Some(match arity {
        0 => |_vm, _a| Ok(Value::object(JStringBuilder::default())),
        1 => |_vm, a| {
            let init = match &a[0] {
                Value::Int(n) if *n < 0 => {
                    return Err(Throw::new(
                        "java.lang.NegativeArraySizeException",
                        n.to_string(),
                    ))
                }
                Value::Int(_) => String::new(),
                Value::Null => return Err(npe()),
                v => v.java_to_string(),
            };
            Ok(Value::object(JStringBuilder {
                buf: RefCell::new(init),
            }))
        },
        _ => return None,
    })
}

impl JObject for JStringBuilder {
    fn class_name(&self) -> &str {
        "java.lang.StringBuilder"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn instance_of(&self, class: &str) -> bool {
        matches!(
            class,
            "java.lang.StringBuilder"
                | "java.lang.CharSequence"
                | "java.lang.Object"
                | "java.lang.Appendable"
        )
    }

    fn java_to_string(&self) -> String {
        self.buf.borrow().clone()
    }

    fn call(
        &self,
        vm: &mut Vm,
        this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        let r = (|| -> Result<Value, Throw> {
            Ok(match (name, args.len()) {
                ("append", 1) => {
                    args[0].write_java(&mut self.buf.borrow_mut(), 0);
                    this.clone()
                }
                ("append", 3) => {
                    let s = args[0].java_to_string();
                    let part = substring(&s, int_arg(&args[1])?, int_arg(&args[2])?)?;
                    self.buf.borrow_mut().push_str(&part);
                    this.clone()
                }
                ("appendCodePoint", 1) => {
                    let cp = int_arg(&args[0])?;
                    self.buf
                        .borrow_mut()
                        .push(char::from_u32(cp as u32).unwrap_or('\u{FFFD}'));
                    this.clone()
                }
                ("length", 0) => Value::Int(len16(&self.buf.borrow()) as i32),
                ("toString", 0) => Value::string(self.buf.borrow().clone()),
                ("reverse", 0) => {
                    let r: String = self.buf.borrow().chars().rev().collect();
                    *self.buf.borrow_mut() = r;
                    this.clone()
                }
                ("insert", 2) => {
                    let i = int_arg(&args[0])?;
                    let s = self.buf.borrow().clone();
                    let n = len16(&s) as i32;
                    if i < 0 || i > n {
                        return Err(sioobe(format!("offset {i}, length {n}")));
                    }
                    let new = format!(
                        "{}{}{}",
                        substring(&s, 0, i)?,
                        args[1].java_to_string(),
                        substring(&s, i, n)?
                    );
                    *self.buf.borrow_mut() = new;
                    this.clone()
                }
                ("delete", 2) => {
                    let s = self.buf.borrow().clone();
                    let n = len16(&s) as i32;
                    let (a, b) = (int_arg(&args[0])?, int_arg(&args[1])?.min(n));
                    if a < 0 || a > b {
                        return Err(sioobe(format!("start {a}, end {b}, length {n}")));
                    }
                    let new = format!("{}{}", substring(&s, 0, a)?, substring(&s, b, n)?);
                    *self.buf.borrow_mut() = new;
                    this.clone()
                }
                ("deleteCharAt", 1) => {
                    let s = self.buf.borrow().clone();
                    let n = len16(&s) as i32;
                    let a = int_arg(&args[0])?;
                    if a < 0 || a >= n {
                        return Err(sioobe(format!("index {a},length {n}")));
                    }
                    let new = format!("{}{}", substring(&s, 0, a)?, substring(&s, a + 1, n)?);
                    *self.buf.borrow_mut() = new;
                    this.clone()
                }
                ("replace", 3) => {
                    let s = self.buf.borrow().clone();
                    let n = len16(&s) as i32;
                    let (a, b) = (int_arg(&args[0])?, int_arg(&args[1])?.min(n));
                    let new = format!(
                        "{}{}{}",
                        substring(&s, 0, a)?,
                        str_arg(&args[2])?,
                        substring(&s, b, n)?
                    );
                    *self.buf.borrow_mut() = new;
                    this.clone()
                }
                ("setLength", 1) => {
                    let n = int_arg(&args[0])?;
                    let s = self.buf.borrow().clone();
                    let len = len16(&s) as i32;
                    let new = if n <= len {
                        substring(&s, 0, n)?
                    } else {
                        let mut t = s;
                        t.extend(std::iter::repeat_n('\0', (n - len) as usize));
                        t
                    };
                    *self.buf.borrow_mut() = new;
                    Value::Null
                }
                ("setCharAt", 2) => {
                    let i = int_arg(&args[0])?;
                    let mut u = utf16(&self.buf.borrow());
                    let c = match &args[1] {
                        Value::Char(c) => *c,
                        _ => return Err(Throw::class_cast("not a char")),
                    };
                    let n = u.len();
                    *u.get_mut(usize::try_from(i).unwrap_or(usize::MAX))
                        .ok_or_else(|| sioobe(format!("index {i},length {n}")))? = c;
                    *self.buf.borrow_mut() = from_utf16(&u);
                    Value::Null
                }
                ("capacity", 0) => Value::Int((len16(&self.buf.borrow()) as i32).max(16)),
                _ => {
                    let s: Rc<str> = Rc::from(self.buf.borrow().as_str());
                    return match call(vm, &s, name, args) {
                        Some(r) if !matches!(name, "equals" | "hashCode") => r,
                        _ => Err(Throw::new(api::UNSUPPORTED, "")),
                    };
                }
            })
        })();
        match r {
            Err(t) if t.class() == api::UNSUPPORTED && t.message().as_deref() == Some("") => None,
            r => Some(r),
        }
    }
}
