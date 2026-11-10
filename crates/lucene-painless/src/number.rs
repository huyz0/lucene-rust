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
    // Rust's `{:e}` is the shortest round-tripping digits.
    let mut sci = format!("{v:e}");
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

/// `Double.parseDouble`: Java's grammar (leading/trailing whitespace,
/// `d`/`f` suffixes, hex floats left out), `None` where Java throws.
pub fn parse_double(s: &str) -> Option<f64> {
    let t = s.trim_matches(|c: char| c <= ' ');
    let t = t.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(t);
    match t {
        "NaN" | "+NaN" | "-NaN" => return Some(f64::NAN),
        "Infinity" | "+Infinity" => return Some(f64::INFINITY),
        "-Infinity" => return Some(f64::NEG_INFINITY),
        _ => {}
    }
    if t.is_empty() || t.contains(|c: char| c.is_ascii_alphabetic() && c != 'e' && c != 'E') {
        return None;
    }
    t.parse::<f64>().ok()
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
