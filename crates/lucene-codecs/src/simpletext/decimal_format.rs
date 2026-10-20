//! The `java.text.DecimalFormat` the SimpleText doc values format formats
//! and parses its fixed-width numbers with: `new DecimalFormat("000…",
//! DecimalFormatSymbols(Locale.ROOT))`, a pattern of zeros only.
//!
//! Formatting zero-pads to the pattern's width (a longer number keeps all
//! its digits; a negative one is `-` and the padded magnitude). Parsing is
//! `DecimalFormat.parse(String)`'s lenient grammar: an optional `-`, digits
//! (leading zeros skipped), an optional `.` and fraction, an optional `E`
//! exponent with an optional `-` (`E+3` is `1`: the `E` is dropped), and
//! anything after the number ignored -- `,` too, since a zeros-only pattern
//! turns grouping off; `NaN` and `∞` are the special values. (The tests
//! check every probe against what JDK 21 gave for it, dumped by
//! `fixtures/src/GenSimpleTextDecimalFormat.java`.)
//! The result is kept as Java's `DigitList` holds it -- significant digits
//! and a decimal exponent -- so each caller converts it the way Java does:
//! `BigDecimal.toBigIntegerExact` ([`Parsed::to_long_exact_wrapping`]),
//! `Number.intValue` ([`Parsed::int_value`]) or `Number.longValue`
//! ([`Parsed::long_value`]).
//!
//! What differs from Java: digits are ASCII only (Java's `Character.digit`
//! also takes other Unicode decimal digits), and a pattern other than one or
//! more `0`s is refused ([`check_pattern`]) where Java would build some
//! other format from it.

use super::{corrupt, Result};

/// `new DecimalFormat(pattern, ROOT)` for the patterns SimpleText writes.
///
/// # Errors
/// `Corrupted` for an empty pattern or one holding anything but `0`.
pub fn check_pattern(pattern: &str) -> Result<()> {
    if !pattern.is_empty() && pattern.bytes().all(|b| b == b'0') {
        Ok(())
    } else {
        Err(corrupt(format!(
            "unsupported DecimalFormat pattern \"{pattern}\""
        )))
    }
}

/// `new DecimalFormat("0" * width, ROOT).format(v)`.
pub fn format(v: i128, width: usize) -> String {
    let digits = v.unsigned_abs().to_string();
    let mut s = String::with_capacity(width.max(digits.len()).saturating_add(1));
    if v < 0 {
        s.push('-');
    }
    for _ in digits.len()..width {
        s.push('0');
    }
    s.push_str(&digits);
    s
}

/// What `DecimalFormat.parse` read.
#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    /// `NaN` (a `Double`).
    NaN,
    /// `∞` or `-∞` (a `Double`); `true` when negative.
    Infinite(bool),
    /// `±0.d₁d₂… × 10^decimal_at`, as `DigitList` holds it.
    Digits {
        /// The `-` prefix.
        negative: bool,
        /// Significant digits (ASCII), no leading zeros.
        digits: Vec<u8>,
        /// `DigitList.decimalAt`.
        decimal_at: i32,
    },
}

/// `DecimalFormat.parse(text)` (lenient, `parsePosition` 0).
///
/// # Errors
/// `Corrupted` when no number starts the text (Java's `ParseException`).
pub fn parse(text: &str) -> Result<Parsed> {
    // Java: DecimalFormat.parse
    if text.starts_with("NaN") {
        return Ok(Parsed::NaN);
    }
    let (negative, rest) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    if rest.starts_with('\u{221e}') {
        return Ok(Parsed::Infinite(negative));
    }
    let (digits, decimal_at, saw_digit) = subparse_number(rest.as_bytes(), false)?;
    if !saw_digit {
        return Err(corrupt(format!("Unparseable number: \"{text}\"")));
    }
    Ok(Parsed::Digits {
        negative,
        digits,
        decimal_at,
    })
}

/// `DecimalFormat.subparseNumber`: the digits, `decimalAt` and whether a
/// digit was seen. `exponent` parses an exponent's digits (no `.` or
/// nested `E`).
fn subparse_number(text: &[u8], exponent: bool) -> Result<(Vec<u8>, i32, bool)> {
    // Java: DecimalFormat.subparseNumber
    let mut digits: Vec<u8> = Vec::new();
    let mut decimal_at: i32 = 0;
    let mut digit_count: i32 = 0;
    let mut saw_decimal = false;
    let mut saw_digit = false;
    let mut exp: Option<i32> = None;
    let mut i = 0usize;
    while let Some(&c) = text.get(i) {
        if c == b'0' {
            saw_digit = true;
            if digits.is_empty() {
                if saw_decimal {
                    decimal_at = decimal_at.wrapping_sub(1);
                }
            } else {
                digit_count = digit_count.saturating_add(1);
                digits.push(c);
            }
        } else if c.is_ascii_digit() {
            saw_digit = true;
            digit_count = digit_count.saturating_add(1);
            digits.push(c);
        } else if !exponent && c == b'.' {
            if saw_decimal {
                break;
            }
            decimal_at = digit_count;
            saw_decimal = true;
        } else if !exponent && c == b'E' {
            exp = parse_exponent(text.get(i.saturating_add(1)..).unwrap_or_default())?;
            break;
        } else {
            break;
        }
        i = i.saturating_add(1);
    }
    if !saw_decimal {
        decimal_at = digit_count;
    }
    if let Some(e) = exp {
        decimal_at = decimal_at.wrapping_add(e);
    }
    Ok((digits, decimal_at, saw_digit))
}

/// The exponent after `E`: `subparse(text, pos, "", "-", exponentDigits,
/// true, stat)` and, if it fits a `long`, `(int)` of it (negated for `-`);
/// `None` when no exponent parses (the `E` is then ignored).
fn parse_exponent(text: &[u8]) -> Result<Option<i32>> {
    let (negative, rest) = match text.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, text),
    };
    let (digits, decimal_at, saw_digit) = subparse_number(rest, true)?;
    if !saw_digit {
        return Ok(None);
    }
    let Some(v) = fits_long(negative, &digits, decimal_at, true) else {
        return Ok(None);
    };
    // `(int) exponentDigits.getLong()`, then negated: both wrap in Java.
    let e = v as i32;
    Ok(Some(if negative { e.wrapping_neg() } else { e }))
}

/// `DigitList.fitsIntoLong(isPositive, ignoreNegativeZero)` and, when it
/// fits, `getLong()` (the magnitude, as a wrapping `long`: `MIN_VALUE` for
/// 9223372036854775808).
fn fits_long(
    negative: bool,
    digits: &[u8],
    decimal_at: i32,
    ignore_negative_zero: bool,
) -> Option<i64> {
    let mut count = digits.len();
    while count > 0 && digits.get(count.wrapping_sub(1)) == Some(&b'0') {
        count = count.wrapping_sub(1);
    }
    if count == 0 {
        return (!negative || ignore_negative_zero).then_some(0);
    }
    let at = usize::try_from(decimal_at).ok()?;
    if at < count || at > 19 {
        return None;
    }
    let mut v: u64 = 0;
    for &d in digits.get(..count)? {
        v = v
            .wrapping_mul(10)
            .wrapping_add(u64::from(d.wrapping_sub(b'0')));
    }
    for _ in count..at {
        v = v.wrapping_mul(10);
    }
    // 19 digits may still overflow: Long.MIN_VALUE's magnitude fits only a
    // negative number. (Fewer than 19 digits always fit.)
    if at == 19 {
        let limit = 1u64 << 63;
        let digits_19: Vec<u8> = digits
            .iter()
            .copied()
            .chain(std::iter::repeat(b'0'))
            .take(19)
            .collect();
        if digits_19.as_slice() > b"9223372036854775808".as_slice() {
            return None;
        }
        if v == limit && !negative {
            return None;
        }
    }
    Some(v as i64)
}

impl Parsed {
    /// `((BigDecimal) parsed).toBigIntegerExact()`, then `BigInteger
    /// .valueOf(origin).add(it).longValue()`: the low 64 bits of `origin +
    /// value`.
    ///
    /// # Errors
    /// `Corrupted` for `NaN`/`∞` (Java's `ClassCastException`) and for a
    /// value with a fraction (its `ArithmeticException`).
    pub fn to_long_exact_wrapping(&self, origin: i64) -> Result<i64> {
        let Parsed::Digits {
            negative,
            digits,
            decimal_at,
        } = self
        else {
            return Err(corrupt("failed to parse BigDecimal value"));
        };
        let mut count = digits.len();
        while count > 0 && digits.get(count.wrapping_sub(1)) == Some(&b'0') {
            count = count.wrapping_sub(1);
        }
        let mut v: u64 = 0;
        if count > 0 {
            let scale =
                i64::from(*decimal_at).saturating_sub(i64::try_from(count).unwrap_or(i64::MAX));
            if scale < 0 {
                return Err(corrupt("Rounding necessary"));
            }
            for &d in digits.get(..count).unwrap_or_default() {
                v = v
                    .wrapping_mul(10)
                    .wrapping_add(u64::from(d.wrapping_sub(b'0')));
            }
            // 10^64 is a multiple of 2^64: past that the low bits are zero.
            for _ in 0..scale.min(64) {
                v = v.wrapping_mul(10);
            }
        }
        let v = if *negative { v.wrapping_neg() } else { v };
        Ok(origin.wrapping_add(v as i64))
    }

    /// What `DecimalFormat.parse` returns without `setParseBigDecimal`: a
    /// `Long` when the value is integral and fits, else a `Double`.
    fn number(&self) -> std::result::Result<i64, f64> {
        match self {
            Parsed::NaN => Err(f64::NAN),
            Parsed::Infinite(negative) => Err(if *negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }),
            Parsed::Digits {
                negative,
                digits,
                decimal_at,
            } => match fits_long(*negative, digits, *decimal_at, false) {
                // `longResult = -longResult` (Long.MIN_VALUE stays itself).
                Some(v) if *negative => Ok(v.wrapping_neg()),
                Some(v) => Ok(v),
                None => {
                    // `DigitList.getDouble()`: Double.parseDouble(".digits" + "E" + decimalAt).
                    let text = format!(
                        "0.{}e{decimal_at}",
                        String::from_utf8_lossy(if digits.is_empty() { b"0" } else { digits })
                    );
                    let d = text.parse::<f64>().unwrap_or(0.0);
                    Err(if *negative { -d } else { d })
                }
            },
        }
    }

    /// `Number.intValue()` of the parse: a `Long` narrowed (wrapping), a
    /// `Double` truncated and saturated (`NaN` is 0).
    pub fn int_value(&self) -> i32 {
        match self.number() {
            Ok(l) => l as i32,
            Err(d) => d as i32,
        }
    }

    /// `Number.longValue()` of the parse: a `Double` truncated and
    /// saturated (`NaN` is 0).
    pub fn long_value(&self) -> i64 {
        match self.number() {
            Ok(l) => l,
            Err(d) => d as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digits(text: &str) -> (bool, String, i32) {
        match parse(text).unwrap() {
            Parsed::Digits {
                negative,
                digits,
                decimal_at,
            } => (negative, String::from_utf8(digits).unwrap(), decimal_at),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn formats_zero_padded() {
        assert_eq!(format(5, 3), "005");
        assert_eq!(format(0, 1), "0");
        assert_eq!(format(12345, 3), "12345");
        assert_eq!(format(-5, 3), "-005");
        assert_eq!(format(u64::MAX as i128, 20), "18446744073709551615");
        assert_eq!(format(i128::from(i64::MIN), 0), "-9223372036854775808");
    }

    #[test]
    fn patterns_of_zeros_only() {
        assert!(check_pattern("0").is_ok());
        assert!(check_pattern("0000").is_ok());
        assert!(check_pattern("").is_err());
        assert!(check_pattern("00#").is_err());
        assert!(check_pattern("XX").is_err());
    }

    #[test]
    fn parses_digit_lists() {
        assert_eq!(digits("000123"), (false, "123".into(), 3));
        assert_eq!(digits("-0120"), (true, "120".into(), 3));
        assert_eq!(digits("0"), (false, String::new(), 0));
        // Grouping is off: `,` ends the number, as does a second `.`.
        assert_eq!(digits("1,234"), (false, "1".into(), 1));
        assert_eq!(digits("1,234.5"), (false, "1".into(), 1));
        assert_eq!(digits(".5"), (false, "5".into(), 0));
        assert_eq!(digits("12 "), (false, "12".into(), 2));
        assert_eq!(digits("12.50"), (false, "1250".into(), 2));
        assert_eq!(digits("0.05"), (false, "5".into(), -1));
        assert_eq!(digits("12.5.6"), (false, "125".into(), 2));
        assert_eq!(digits("1.2,3"), (false, "12".into(), 1));
        assert_eq!(digits("1E3"), (false, "1".into(), 4));
        assert_eq!(digits("15E-1"), (false, "15".into(), 1));
        // No exponent digits, or a `+`: the `E` is not part of the number.
        assert_eq!(digits("1E"), (false, "1".into(), 1));
        assert_eq!(digits("1E+3"), (false, "1".into(), 1));
        // An exponent too large for a long is ignored.
        assert_eq!(digits("1E99999999999999999999"), (false, "1".into(), 1));
        assert_eq!(digits("7 trailing"), (false, "7".into(), 1));
        assert_eq!(parse("NaN").unwrap(), Parsed::NaN);
        assert_eq!(parse("\u{221e}").unwrap(), Parsed::Infinite(false));
        assert_eq!(parse("-\u{221e}").unwrap(), Parsed::Infinite(true));
        for bad in ["", "-", "x1", " 1", "+1", ".", ","] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn exact_big_decimals_wrap_to_a_long() {
        let exact = |t: &str, origin: i64| parse(t).unwrap().to_long_exact_wrapping(origin);
        assert_eq!(exact("0005", 10).unwrap(), 15);
        assert_eq!(exact("0", -3).unwrap(), -3);
        assert_eq!(exact("1.00", 0).unwrap(), 1);
        assert_eq!(exact("100E-2", 0).unwrap(), 1);
        assert_eq!(exact("-7", 0).unwrap(), -7);
        // Long.MIN_VALUE + (2^64 - 1) = Long.MAX_VALUE.
        assert_eq!(exact("18446744073709551615", i64::MIN).unwrap(), i64::MAX);
        // Past 2^64 only the low bits remain.
        assert_eq!(exact("18446744073709551616", 0).unwrap(), 0);
        assert_eq!(exact("1E64", 5).unwrap(), 5);
        assert_eq!(exact("1E70", 5).unwrap(), 5);
        assert!(exact("1.5", 0).is_err());
        assert!(exact("0.05", 0).is_err());
        assert!(parse("NaN").unwrap().to_long_exact_wrapping(0).is_err());
        assert!(parse("\u{221e}")
            .unwrap()
            .to_long_exact_wrapping(0)
            .is_err());
    }

    /// Every probe of `GenSimpleTextDecimalFormat` against what JDK 21's
    /// `DecimalFormat("000", ROOT)` gave for it.
    #[test]
    fn matches_java_on_every_probe() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/codecs_simpletext/decimal_format.tsv"
        );
        let tsv = std::fs::read_to_string(path).unwrap();
        let mut n = 0;
        for line in tsv.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let bytes: Vec<u8> = (0..cols[0].len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&cols[0][i..i + 2], 16).unwrap())
                .collect();
            let probe = String::from_utf8(bytes).unwrap();
            let got: Vec<String> = match parse(&probe) {
                Err(_) => vec!["PARSE".into(); 4],
                Ok(p) => {
                    let exact = |origin| {
                        p.to_long_exact_wrapping(origin)
                            .map_or_else(|_| "ERR".to_string(), |v| v.to_string())
                    };
                    vec![
                        exact(0),
                        exact(i64::MIN),
                        p.int_value().to_string(),
                        p.long_value().to_string(),
                    ]
                }
            };
            assert_eq!(got, cols[1..], "{probe:?}");
            n += 1;
        }
        assert!(n > 40, "{n} probes");
    }

    #[test]
    fn number_values_as_java_narrows_them() {
        let int = |t: &str| parse(t).unwrap().int_value();
        let long = |t: &str| parse(t).unwrap().long_value();
        assert_eq!(int("0042"), 42);
        assert_eq!(int("-3"), -3);
        assert_eq!(int("-0"), 0);
        assert_eq!(int("4294967297"), 1);
        assert_eq!(int("1.9"), 1);
        assert_eq!(int("-1.9"), -1);
        assert_eq!(int("1E30"), i32::MAX);
        assert_eq!(int("NaN"), 0);
        assert_eq!(int("-\u{221e}"), i32::MIN);
        assert_eq!(long("9223372036854775807"), i64::MAX);
        assert_eq!(long("-9223372036854775808"), i64::MIN);
        assert_eq!(long("9223372036854775808"), i64::MAX);
        assert_eq!(long("-9223372036854775809"), i64::MIN);
        assert_eq!(long("92233720368547758070"), i64::MAX);
        assert_eq!(long("\u{221e}"), i64::MAX);
        assert_eq!(long("0.5"), 0);
    }
}
