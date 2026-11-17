//! Java's number semantics: conversions (`d2i` saturates, `NaN` to 0),
//! `Float.toString`/`Double.toString`, and the bit views `equals` and
//! `hashCode` use.

/// `Float.floatToIntBits`: every NaN is the canonical one.
pub fn float_to_int_bits(v: f32) -> i32 {
    if v.is_nan() {
        0x7fc0_0000
    } else {
        v.to_bits() as i32
    }
}

/// `Double.doubleToLongBits`.
pub fn double_to_long_bits(v: f64) -> i64 {
    if v.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        v.to_bits() as i64
    }
}

/// `(int) d`: Rust's saturating `as` is Java's `d2i`/`f2i`.
pub fn d2i(v: f64) -> i32 {
    v as i32
}

/// `(long) d`.
pub fn d2l(v: f64) -> i64 {
    v as i64
}

/// `Float.toString`.
pub fn float_to_string(v: f32) -> String {
    lucene_analysis::attributes::java_float_to_string(v)
}

/// `Double.toString` (JDK 19 and later): the shortest decimal that reads
/// back as `v`, at least two digits, plain for `10^-3 <= |v| < 10^7`,
/// otherwise `d.dddEn`.
pub fn double_to_string(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` is the shortest round-tripping digits, but on a tie
    // between two of them it rounds up where Java takes the even one: the
    // exact decimal expansion rounded to that many digits (half to even)
    // is Java's choice whenever it reads back as `v`.
    let mut sci = format!("{v:e}");
    let shortest_digits = sci
        .split_once('e')
        .map_or(0, |(m, _)| m.chars().filter(char::is_ascii_digit).count());
    if shortest_digits > 1 {
        let exact = format!("{v:.prec$e}", prec = shortest_digits - 1);
        if exact.parse::<f64>().is_ok_and(|r| r == v) {
            sci = exact;
        }
    }
    let (mant, exp) = sci
        .split_once('e')
        .map(|(m, e)| (m.to_string(), e.parse::<i32>().unwrap_or(0)))
        .unwrap_or_default();
    let neg = mant.starts_with('-');
    let mut digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    if digits.len() == 1 {
        // Java prints at least two digits: the two-digit decimal closest to `v`.
        sci = format!("{v:.1e}");
        let (m2, e2) = sci.split_once('e').unwrap_or((&sci, "0"));
        let d2: String = m2.chars().filter(char::is_ascii_digit).collect();
        let e2: i32 = e2.parse().unwrap_or(exp);
        return layout(neg, d2.trim_end_matches('0').to_string(), e2);
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    layout(neg, digits, exp)
}

/// Lays out significant `digits` (`d1 d2 ...` = `d1.d2... * 10^exp`) as Java does.
fn layout(neg: bool, digits: String, exp: i32) -> String {
    let digits = if digits.is_empty() {
        "0".to_string()
    } else {
        digits
    };
    let mut out = String::with_capacity(digits.len() + 8);
    if neg {
        out.push('-');
    }
    if (-3..7).contains(&exp) {
        if exp < 0 {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let int_len = (exp + 1) as usize;
            if digits.len() <= int_len {
                out.push_str(&digits);
                for _ in digits.len()..int_len {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        out.push('.');
        if digits.len() > 1 {
            out.push_str(&digits[1..]);
        } else {
            out.push('0');
        }
        out.push('E');
        out.push_str(&exp.to_string());
    }
    out
}

/// `Integer.parseInt(s, radix)`: `None` where Java throws `NumberFormatException`.
pub fn parse_long_radix(s: &str, radix: u32, min: i64, max: i64) -> Option<i64> {
    let (neg, body) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if body.is_empty() {
        return None;
    }
    let mut v: i128 = 0;
    for c in body.chars() {
        let d = c.to_digit(radix)?;
        v = v * i128::from(radix) + i128::from(d);
        if v > i128::from(max) + 1 {
            return None;
        }
    }
    let v = if neg { -v } else { v };
    if v < i128::from(min) || v > i128::from(max) {
        return None;
    }
    Some(v as i64)
}

/// The numeric text of a Java float string: trimmed, one `d`/`f` suffix
/// dropped, `None` where Java's grammar rejects it. The special values and
/// hex floats are handled by the callers.
fn decimal_text(t: &str) -> Option<&str> {
    let t = t.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(t);
    if t.is_empty() || t.contains(|c: char| c.is_ascii_alphabetic() && c != 'e' && c != 'E') {
        return None;
    }
    Some(t)
}

fn special(t: &str) -> Option<f64> {
    match t {
        "NaN" | "+NaN" | "-NaN" => Some(f64::NAN),
        "Infinity" | "+Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// A hex float, `[+-]0x<hex>[.<hex>]p[+-]<digits>[fFdD]`, rounded to nearest.
fn parse_hex_float(t: &str) -> Option<f64> {
    let (negative, t) = match t.as_bytes().first()? {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))?;
    let t = t.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(t);
    let (mantissa, exponent) = t.split_once(['p', 'P'])?;
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    let exponent: i32 = {
        let digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        exponent.parse().unwrap_or(if exponent.starts_with('-') {
            i32::MIN / 2
        } else {
            i32::MAX / 2
        })
    };
    let (mut m, mut sticky, mut scale) = (0u128, false, 0i32);
    for (i, c) in int.chars().chain(frac.chars()).enumerate() {
        let d = u128::from(c.to_digit(16)?);
        if m >> 120 == 0 {
            m = (m << 4) | d;
            if i >= int.len() {
                scale -= 4;
            }
        } else {
            sticky |= d != 0;
            if i < int.len() {
                scale += 4;
            }
        }
    }
    if m == 0 {
        return Some(if negative { -0.0 } else { 0.0 });
    }
    // Fold what did not fit into the lowest bit so rounding sees it.
    if sticky {
        m |= 1;
    }
    let v = m as f64;
    let mut e = i64::from(exponent) + i64::from(scale);
    let mut v = v;
    while e > 0 {
        let step = e.min(1000);
        v *= 2f64.powi(step as i32);
        e -= step;
    }
    while e < 0 {
        let step = (-e).min(1000);
        v *= 2f64.powi(-(step as i32));
        e += step;
    }
    Some(if negative { -v } else { v })
}

/// `Double.parseDouble`: Java's grammar (surrounding whitespace, a `d`/`f`
/// suffix, hex floats), `None` where Java throws.
pub fn parse_double(s: &str) -> Option<f64> {
    let t = s.trim_matches(|c: char| c <= ' ');
    if let Some(v) = special(t) {
        return Some(v);
    }
    if let Some(v) = parse_hex_float(t) {
        return Some(v);
    }
    decimal_text(t)?.parse::<f64>().ok()
}

/// `Float.parseFloat`: the grammar of [`parse_double`], rounded once to `f32`.
pub fn parse_float(s: &str) -> Option<f32> {
    let t = s.trim_matches(|c: char| c <= ' ');
    if let Some(v) = special(t) {
        return Some(v as f32);
    }
    if let Some(v) = parse_hex_float(t) {
        return Some(v as f32);
    }
    decimal_text(t)?.parse::<f32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_print_as_java() {
        for (v, s) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (100.0, "100.0"),
            (1e7, "1.0E7"),
            (1.5e-4, "1.5E-4"),
            (0.001, "0.001"),
            (123456.789, "123456.789"),
            (-2.5, "-2.5"),
            (f64::MIN_POSITIVE, "2.2250738585072014E-308"),
            (5e-324, "4.9E-324"),
            (1e21, "1.0E21"),
            (9999999.0, "9999999.0"),
            (12345678.9, "1.23456789E7"),
        ] {
            assert_eq!(double_to_string(v), s, "{v}");
        }
    }
}
