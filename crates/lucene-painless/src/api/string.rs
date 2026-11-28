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
        return Err(sioobe(format!(
            "Range [{a}, {b}) out of bounds for length {len}"
        )));
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

/// `indexOf` in UTF-16 units from `from`: the first match, if any.
fn index_of(s: &str, needle: &str, from: i32) -> Option<usize> {
    let (h, n) = (utf16(s), utf16(needle));
    let from = from.max(0) as usize;
    if n.is_empty() {
        return Some(from.min(h.len()));
    }
    if n.len() > h.len() {
        return None;
    }
    (from..=h.len() - n.len()).find(|&i| h[i..i + n.len()] == n[..])
}

/// `lastIndexOf` in UTF-16 units, searching back from `from`.
fn last_index_of(s: &str, needle: &str, from: i32) -> Option<usize> {
    let (h, n) = (utf16(s), utf16(needle));
    if n.len() > h.len() || from < 0 {
        return None;
    }
    let start = (from as usize).min(h.len() - n.len());
    (0..=start).rev().find(|&i| h[i..i + n.len()] == n[..])
}

/// Java's `indexOf` result: the position, or `-1` for "not found".
fn java_index(found: Option<usize>) -> Value {
    Value::Int(found.map_or(-1, |i| i as i32))
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
            ("indexOf", 1) => java_index(index_of(s, &str_arg(&args[0])?, 0)),
            ("indexOf", 2) => java_index(index_of(s, &str_arg(&args[0])?, int_arg(&args[1])?)),
            ("lastIndexOf", 1) => java_index(last_index_of(s, &str_arg(&args[0])?, i32::MAX)),
            ("lastIndexOf", 2) => {
                java_index(last_index_of(s, &str_arg(&args[0])?, int_arg(&args[1])?))
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
            ("chars", 0) => super::stream::of_ints(
                utf16(s)
                    .into_iter()
                    .map(|c| Value::Int(i32::from(c)))
                    .collect(),
            ),
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
                let b = base64_decode(s).map_err(Throw::illegal_arg)?;
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

/// `Base64.getDecoder().decode(s)` with the JDK's checks and messages: the
/// unpadded form is accepted, anything else malformed throws.
pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let src: Vec<u32> = s.chars().map(|c| c as u32).collect();
    let n = src.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n < 2 {
        return Err("Input byte[] should at least have 2 bytes for base64 bytes".into());
    }
    let mut out = Vec::with_capacity(n / 4 * 3 + 3);
    let (mut bits, mut shift, mut sp) = (0u32, 18i32, 0usize);
    while sp < n {
        let c = src[sp];
        sp += 1;
        if c == u32::from(b'=') {
            // Java reads the byte after a first `=`; a unit that starts on `=` or
            // lacks its second `=` is malformed.
            let second_missing = shift == 6 && (sp == n || src[sp] != u32::from(b'='));
            if shift == 6 && !second_missing {
                sp += 1;
            }
            if second_missing || shift == 18 {
                return Err("Input byte array has wrong 4-byte ending unit".into());
            }
            break;
        }
        let v = if c < 128 {
            B64.iter().position(|&x| u32::from(x) == c)
        } else {
            None
        }
        .ok_or_else(|| {
            // Java decodes the Latin-1 bytes: a wider char is `?`, a high byte is negative.
            let b = if c > 0xff { 0x3f } else { c as i32 };
            let b = if b > 127 { b - 256 } else { b };
            let sign = if b < 0 { "-" } else { "" };
            format!("Illegal base64 character {sign}{:x}", b.abs())
        })? as u32;
        bits |= v << shift;
        shift -= 6;
        if shift < 0 {
            out.extend([(bits >> 16) as u8, (bits >> 8) as u8, bits as u8]);
            shift = 18;
            bits = 0;
        }
    }
    match shift {
        6 => out.push((bits >> 16) as u8),
        0 => out.extend([(bits >> 16) as u8, (bits >> 8) as u8]),
        12 => return Err("Last unit does not have enough valid bits".into()),
        _ => {}
    }
    if sp < n {
        return Err(format!(
            "Input byte array has incorrect ending byte at {sp}"
        ));
    }
    Ok(out)
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
                    .map(char_string)
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
        ("format", 2) => |_vm, a| Ok(Value::string(format(&str_arg(&a[0])?, &varargs(&a[1])?)?)),
        ("format", 3) => |_vm, a| Ok(Value::string(format(&str_arg(&a[1])?, &varargs(&a[2])?)?)),
        _ => return None,
    })
}

/// The `def[]` a `String.format` takes: a lone value that reaches it through a `def` is
/// not an array, which the JVM refuses.
fn varargs(v: &Value) -> Result<Vec<Value>, Throw> {
    match v {
        Value::Array(a) => Ok(a.borrow().items.clone()),
        Value::Null => Ok(Vec::new()),
        other => {
            let from = match other {
                Value::Bool(_) => "boolean".to_string(),
                Value::Byte(_) => "byte".into(),
                Value::Short(_) => "short".into(),
                Value::Char(_) => "char".into(),
                Value::Int(_) => "int".into(),
                Value::Long(_) => "long".into(),
                Value::Float(_) => "float".into(),
                Value::Double(_) => "double".into(),
                o => format!("class {}", o.class_name()),
            };
            Err(Throw::class_cast(format!(
                "Cannot cast from: {from} to class [Ljava.lang.Object;"
            )))
        }
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

/// A parsed `%[index$][flags][width][.precision]conversion`.
#[derive(Default)]
struct Spec {
    /// `1$` as 0, `<` as `usize::MAX`.
    index: Option<usize>,
    flags: String,
    width: Option<usize>,
    prec: Option<usize>,
    conv: char,
}

impl Spec {
    fn has(&self, f: char) -> bool {
        self.flags.contains(f)
    }
}

fn format_throw(class: &str, message: impl Into<String>) -> Throw {
    Throw::new(&format!("java.util.{class}"), message)
}

/// Reads the specifier after a `%`; `text` is its text for the messages.
fn parse_spec(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<(Spec, String), Throw> {
    let mut text = String::new();
    let take_digits = |chars: &mut std::iter::Peekable<std::str::Chars>, text: &mut String| {
        let mut d = String::new();
        while let Some(&c) = chars.peek() {
            if !c.is_ascii_digit() {
                break;
            }
            d.push(c);
            text.push(c);
            chars.next();
        }
        d
    };
    let mut spec = Spec::default();
    // An argument index `n$` or width: digits, then `$` makes them an index.
    let first = take_digits(chars, &mut text);
    let mut width_digits = first.clone();
    if !first.is_empty() && chars.peek() == Some(&'$') {
        chars.next();
        text.push('$');
        spec.index = first.parse::<usize>().ok().map(|i| i.saturating_sub(1));
        width_digits.clear();
    } else if !first.is_empty() && first.starts_with('0') {
        // Leading zeros of a number are the `0` flag.
        let zeros = first.len() - first.trim_start_matches('0').len();
        spec.flags.push_str(&"0".repeat(zeros));
        width_digits = first[zeros..].to_string();
    }
    if spec.index.is_none() && chars.peek() == Some(&'<') {
        chars.next();
        text.push('<');
        spec.index = Some(usize::MAX);
    }
    if width_digits.is_empty() {
        while let Some(&c) = chars.peek() {
            if !"-#+ 0,(".contains(c) {
                break;
            }
            spec.flags.push(c);
            text.push(c);
            chars.next();
        }
        width_digits = take_digits(chars, &mut text);
    }
    spec.width = width_digits.parse().ok();
    if chars.peek() == Some(&'.') {
        chars.next();
        text.push('.');
        let p = take_digits(chars, &mut text);
        spec.prec =
            Some(p.parse().map_err(|_| {
                format_throw("UnknownFormatConversionException", "Conversion = '.'")
            })?);
    }
    spec.conv = chars
        .next()
        .ok_or_else(|| format_throw("UnknownFormatConversionException", "Conversion = '%'"))?;
    text.push(spec.conv);
    Ok((spec, text))
}

/// `java.util.Formatter` for the common conversions.
pub fn format(fmt: &str, args: &[Value]) -> Result<String, Throw> {
    let mut out = String::new();
    let mut chars = fmt.chars().peekable();
    let (mut next_arg, mut last_arg) = (0usize, None::<usize>);
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let (spec, text) = parse_spec(&mut chars)?;
        let conv = spec.conv;
        if "aAtT".contains(conv) {
            return Err(Throw::new(api::UNSUPPORTED, format!("Formatter %{conv}")));
        }
        if !"bBhHsScCdoxXeEfgGn%".contains(conv) {
            return Err(format_throw(
                "UnknownFormatConversionException",
                format!("Conversion = '{conv}'"),
            ));
        }
        if spec.has('-') && spec.width.is_none() || spec.has('0') && spec.width.is_none() {
            return Err(format_throw(
                "MissingFormatWidthException",
                format!("%{text}"),
            ));
        }
        if spec.has('-') && spec.has('0') || spec.has('+') && spec.has(' ') {
            return Err(format_throw(
                "IllegalFormatFlagsException",
                format!("Flags = '{}'", spec.flags),
            ));
        }
        match conv {
            '%' => {
                let body = "%".to_string();
                pad_into(&mut out, &spec, body, false);
                continue;
            }
            'n' => {
                out.push('\n');
                continue;
            }
            _ => {}
        }
        let at = match spec.index {
            Some(usize::MAX) => last_arg,
            Some(i) => Some(i),
            None => {
                next_arg += 1;
                Some(next_arg - 1)
            }
        };
        last_arg = at;
        let arg = at.and_then(|i| args.get(i).cloned()).ok_or_else(|| {
            format_throw(
                "MissingFormatArgumentException",
                format!("Format specifier '%{text}'"),
            )
        })?;
        let body = format_one(&spec, &arg)?;
        let numeric = matches!(conv, 'd' | 'o' | 'x' | 'X' | 'e' | 'E' | 'f' | 'g' | 'G')
            && !matches!(arg, Value::Null)
            && !body.contains("NaN")
            && !body.contains("Infinity");
        pad_into(&mut out, &spec, body, numeric);
    }
    Ok(out)
}

/// `body` padded to the width: left-justified, zero-filled after the sign
/// and radix prefix for a number, or blank-filled.
fn pad_into(out: &mut String, spec: &Spec, body: String, numeric: bool) {
    let pad = spec.width.unwrap_or(0).saturating_sub(body.chars().count());
    if spec.has('-') {
        out.push_str(&body);
        out.extend(std::iter::repeat_n(' ', pad));
    } else if spec.has('0') && numeric {
        let lead = if body.starts_with(['-', '+', ' ', '(']) {
            1
        } else {
            0
        };
        let rest = &body[lead..];
        let prefix = if rest.starts_with("0x") || rest.starts_with("0X") {
            2
        } else {
            0
        };
        out.push_str(&body[..lead + prefix]);
        out.extend(std::iter::repeat_n('0', pad));
        out.push_str(&body[lead + prefix..]);
    } else {
        out.extend(std::iter::repeat_n(' ', pad));
        out.push_str(&body);
    }
}

fn flag_mismatch(conv: char, flag: char) -> Throw {
    format_throw(
        "FormatFlagsConversionMismatchException",
        format!("Conversion = {conv}, Flags = {flag}"),
    )
}

/// The text of one conversion, before the width.
fn format_one(spec: &Spec, arg: &Value) -> Result<String, Throw> {
    let conv = spec.conv;
    let upper = conv.is_ascii_uppercase();
    let text = |s: String| if upper { s.to_uppercase() } else { s };
    let body = match conv {
        's' | 'S' | 'h' | 'H' | 'b' | 'B' => {
            for f in "0+ ,(".chars() {
                if spec.has(f) {
                    return Err(flag_mismatch(conv, f));
                }
            }
            let s = match conv {
                'h' | 'H' => match arg {
                    Value::Null => "null".to_string(),
                    v => format!("{:x}", v.java_hash() as u32),
                },
                'b' | 'B' => match arg {
                    Value::Null => "false".to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => "true".to_string(),
                },
                _ => arg.java_to_string(),
            };
            let s = match spec.prec {
                Some(p) => s.chars().take(p).collect(),
                None => s,
            };
            text(s)
        }
        'c' | 'C' => {
            if let Some(p) = spec.prec {
                return Err(format_throw(
                    "IllegalFormatPrecisionException",
                    p.to_string(),
                ));
            }
            text(match arg {
                Value::Null => "null".to_string(),
                Value::Char(c) => char_string(*c),
                Value::Int(i) => char::from_u32(*i as u32).map(String::from).ok_or_else(|| {
                    format_throw(
                        "IllegalFormatCodePointException",
                        format!("Code point = 0x{:x}", *i),
                    )
                })?,
                o => return Err(illegal_conv(conv, o)),
            })
        }
        'd' | 'o' | 'x' | 'X' => {
            if let Some(p) = spec.prec {
                return Err(format_throw(
                    "IllegalFormatPrecisionException",
                    p.to_string(),
                ));
            }
            if matches!(arg, Value::Null) {
                return Ok(text("null".into()));
            }
            let (v, bits) = match arg {
                Value::Int(i) => (i64::from(*i), 32),
                Value::Long(i) => (*i, 64),
                Value::Short(i) => (i64::from(*i), 16),
                Value::Byte(i) => (i64::from(*i), 8),
                o => return Err(illegal_conv(conv, o)),
            };
            if conv == 'd' {
                if spec.has('#') {
                    return Err(flag_mismatch(conv, '#'));
                }
                let mut s = v.unsigned_abs().to_string();
                if spec.has(',') {
                    s = group(&s);
                }
                signed(spec, v < 0, s)
            } else {
                for f in "+ ,(".chars() {
                    if spec.has(f) {
                        return Err(flag_mismatch(conv, f));
                    }
                }
                // Negative values print as the two's complement of their width.
                let u = (v as u64) & (u64::MAX >> (64 - bits));
                let digits = if conv == 'o' {
                    format!("{u:o}")
                } else {
                    format!("{u:x}")
                };
                let prefix = match (spec.has('#'), conv) {
                    (false, _) => "",
                    (true, 'o') => "0",
                    (true, _) => "0x",
                };
                text(format!("{prefix}{digits}"))
            }
        }
        'f' | 'e' | 'E' | 'g' | 'G' => {
            if matches!(arg, Value::Null) {
                let s: String = "null".chars().take(spec.prec.unwrap_or(4)).collect();
                return Ok(text(s));
            }
            let v = match arg {
                Value::Float(_) | Value::Double(_) => crate::types::as_f64(arg).unwrap_or(0.0),
                o => return Err(illegal_conv(conv, o)),
            };
            if spec.has(',') && matches!(conv, 'e' | 'E') {
                return Err(flag_mismatch(conv, ','));
            }
            let p = spec.prec.unwrap_or(6);
            let mut s = match conv {
                'f' => format_fixed(v, p),
                'e' | 'E' => format_sci(v, p, conv == 'E'),
                _ => format_general(v, p, conv == 'G'),
            };
            let finite = v.is_finite();
            if spec.has(',') && finite {
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
            if spec.has('#') && conv == 'f' && p == 0 && finite {
                s.push('.');
            }
            if v.is_nan() {
                s
            } else {
                let neg = s.starts_with('-');
                signed(spec, neg, s.trim_start_matches('-').to_string())
            }
        }
        _ => {
            return Err(format_throw(
                "UnknownFormatConversionException",
                format!("Conversion = '{conv}'"),
            ))
        }
    };
    Ok(body)
}

/// The sign flags of a number: `-`, `+`, a leading space, or `(` `)`.
fn signed(spec: &Spec, neg: bool, magnitude: String) -> String {
    if neg {
        if spec.has('(') {
            format!("({magnitude})")
        } else {
            format!("-{magnitude}")
        }
    } else if spec.has('+') {
        format!("+{magnitude}")
    } else if spec.has(' ') {
        format!(" {magnitude}")
    } else {
        magnitude
    }
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
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The shortest decimal digits of a finite `v > 0` and the decimal exponent of
/// the first one (`12.5` is `[1, 2, 5]`, 1).
fn decimal_digits(v: f64) -> (Vec<u8>, i32) {
    let plain = to_plain(&crate::number::double_to_string(v));
    let (i, f) = plain.split_once('.').unwrap_or((&plain, ""));
    let all: Vec<u8> = i.bytes().chain(f.bytes()).map(|b| b - b'0').collect();
    let first = all.iter().position(|&d| d != 0).unwrap_or(0);
    let mut digits = all[first..].to_vec();
    while digits.len() > 1 && digits.last() == Some(&0) {
        digits.pop();
    }
    (digits, i.len() as i32 - 1 - first as i32)
}

/// `digits` rounded HALF_UP to `n` significant digits, as Java's `Formatter`
/// does on the shortest representation; the exponent follows a carry.
fn round_significant(digits: &[u8], exp: i32, n: usize) -> (Vec<u8>, i32) {
    let mut out: Vec<u8> = digits.iter().copied().take(n).collect();
    let up = digits.get(n).is_some_and(|&d| d >= 5);
    out.resize(n, 0);
    let mut exp = exp;
    if up {
        let mut k = n;
        loop {
            if k == 0 {
                out.insert(0, 1);
                out.truncate(n);
                exp += 1;
                break;
            }
            k -= 1;
            if out[k] == 9 {
                out[k] = 0;
            } else {
                out[k] += 1;
                break;
            }
        }
    }
    (out, exp)
}

fn non_finite(v: f64) -> Option<String> {
    (!v.is_finite()).then(|| {
        if v.is_nan() {
            "NaN".into()
        } else if v > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        }
    })
}

fn sci_text(digits: &[u8], exp: i32, neg: bool, upper: bool) -> String {
    let mut s = String::new();
    if neg {
        s.push('-');
    }
    s.push((digits[0] + b'0') as char);
    if digits.len() > 1 {
        s.push('.');
        s.extend(digits[1..].iter().map(|d| (d + b'0') as char));
    }
    s.push(if upper { 'E' } else { 'e' });
    s.push(if exp < 0 { '-' } else { '+' });
    s.push_str(&format!("{:02}", exp.abs()));
    s
}

/// `%.Ne`: one digit, `N` more, and a two-digit exponent, rounded HALF_UP.
pub fn format_sci(v: f64, p: usize, upper: bool) -> String {
    if let Some(s) = non_finite(v) {
        return if upper { s.to_uppercase() } else { s };
    }
    let neg = v.is_sign_negative();
    let (mut digits, exp) = if v == 0.0 {
        (vec![0], 0)
    } else {
        let (d, e) = decimal_digits(v.abs());
        round_significant(&d, e, p + 1)
    };
    digits.resize(p + 1, 0);
    sci_text(&digits, exp, neg, upper)
}

/// `%.Ng`: `N` significant digits, fixed notation when the rounded value is
/// in `[1e-4, 10^N)` and scientific otherwise; trailing zeros are kept.
pub fn format_general(v: f64, p: usize, upper: bool) -> String {
    if let Some(s) = non_finite(v) {
        return if upper { s.to_uppercase() } else { s };
    }
    let p = p.max(1);
    if v == 0.0 {
        return format_fixed(v, p - 1);
    }
    let (d, e) = decimal_digits(v.abs());
    let (digits, exp) = round_significant(&d, e, p);
    if exp < -4 || exp >= p as i32 {
        return sci_text(&digits, exp, v.is_sign_negative(), upper);
    }
    format_fixed(v, (p as i32 - 1 - exp) as usize)
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
    // Java signs by the value, so `-0.001` prints `-0.00`.
    let neg = v.is_sign_negative();
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
                        return Err(sioobe(format!(
                            "Range [{i}, {n}) out of bounds for length {n}"
                        )));
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
                        return Err(sioobe(format!(
                            "Range [{a}, {b}) out of bounds for length {n}"
                        )));
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
                        return Err(sioobe(format!("Index {a} out of bounds for length {n}")));
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
                        .ok_or_else(|| {
                            sioobe(format!("Index {i} out of bounds for length {n}"))
                        })? = c;
                    *self.buf.borrow_mut() = from_utf16(&u);
                    Value::Null
                }
                ("compareTo", 1) => match &args[0] {
                    Value::Object(o) if o.as_any().is::<JStringBuilder>() => {
                        Value::Int(java_compare(&self.buf.borrow(), &o.java_to_string()))
                    }
                    Value::Null => {
                        return Err(Throw::npe(
                            "Cannot read field \"value\" because \"another\" is null",
                        ))
                    }
                    other => {
                        return Err(Throw::cast_failure(
                            &other.class_name(),
                            "java.lang.StringBuilder",
                        ))
                    }
                },
                // `AbstractStringBuilder`'s reads, as `String`'s.
                ("indexOf" | "lastIndexOf", 1 | 2) => {
                    if !matches!(args[0], Value::Str(_)) {
                        return Err(Throw::cast_failure(
                            &args[0].class_name(),
                            "java.lang.String",
                        ));
                    }
                    let s: Rc<str> = Rc::from(self.buf.borrow().as_str());
                    return call(vm, &s, name, args)
                        .unwrap_or_else(|| Err(Throw::new(api::UNSUPPORTED, "")));
                }
                ("charAt", 1)
                | ("codePointAt", 1)
                | ("codePointBefore", 1)
                | ("codePointCount", 2)
                | ("offsetByCodePoints", 2)
                | ("substring", 1 | 2)
                | ("subSequence", 2)
                | ("chars" | "codePoints", 0) => {
                    let s: Rc<str> = Rc::from(self.buf.borrow().as_str());
                    return call(vm, &s, name, args)
                        .unwrap_or_else(|| Err(Throw::new(api::UNSUPPORTED, "")));
                }
                // Everything else, `capacity()` among them (its growth is
                // not modelled), is not served.
                _ => return Err(Throw::new(api::UNSUPPORTED, "")),
            })
        })();
        match r {
            Err(t) if t.class() == api::UNSUPPORTED && t.message().as_deref() == Some("") => None,
            r => Some(r),
        }
    }
}

#[cfg(test)]
mod format_tests;
#[cfg(test)]
mod java_diff_tests;
#[cfg(test)]
mod tests;
