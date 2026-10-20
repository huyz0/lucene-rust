//! `java.text.DecimalFormat` as `NumberFormat.getNumberInstance(locale)`
//! (`#,##0.###`) and `NumberFormat.getIntegerInstance(locale)` (`#,##0`,
//! parse integer only) return it: `parse` and `format` of a `long` or a
//! `double`, in a [`DateLocale`] of the generated table, whose records hold
//! the instances' symbols (`tools/GenDateLocales.java` asserts that every
//! locale's two instances have these patterns and differ from its year
//! field's format only in them).
//!
//! The behaviour is written from `DecimalFormat`'s specification and black-
//! box probes of JDK 25 (`queryparser_flexible/numbers.tsv` is JDK 21's
//! answer in the locales whose symbols the two agree on; the unit tests pin
//! JDK 25's where the parse differs); no JDK code is copied
//! (`docs/licences.md`).
//!
//! `parse(text, position)`, over UTF-16 units, from the position:
//!
//! - The locale's NaN symbol (no prefix) is `Double.NaN`.
//! - Otherwise an optional negative prefix (the locale's own only: `-`,
//!   `\u{200e}\u{2212}`, `\u{2212}`, ...; a positive one is not taken), then the infinity
//!   symbol (`±Infinity`), or the number: decimal digits (any BMP `Nd`
//!   `char`, `Character.digit`), grouping separators (the locale's own
//!   `char` only, anywhere in the integer part: `1,,2` is 12, `,1` is 1), one
//!   decimal separator, then optionally the exponent separator, an optional
//!   negative prefix and digits (an exponent with no digit is not taken).
//!   The number ends at anything else, at a second decimal separator, at a
//!   grouping separator after the decimal one, and after an exponent. A
//!   grouping separator not followed by a digit is given back (`1,` ends at
//!   1), even when an exponent follows (`1,E3` is 1000, ending at 1).
//! - The integer instance stops at the decimal separator, without giving a
//!   grouping separator back (JDK 25: `1,.5` ends at 2; JDK 21 at 1), and
//!   after a grouping separator a digit right after it counts as the digit
//!   the parse needs (JDK 25: `,.5` is 0 but `.5` fails; JDK 21 fails both).
//! - No digit read: the parse fails.
//! - The value: a `Long` when integral and within `long`'s range (`1.000`,
//!   `0.5E1`), else a `Double` (`-0` is `-0.0` but for the integer
//!   instance's `0`; an exponent past `int` saturates, JDK 25's: `1E99999999999`
//!   is `Infinity`).
//!
//! `format`: the negative prefix for a negative value (`-0.0` and values
//! rounding to zero included), the integer digits grouped every grouping
//! size from the right, then at most three fraction digits (none for the
//! integer instance) rounded half-even on the exact value, trailing zeros
//! dropped; digits in the locale's zero digit's run; NaN and infinity as
//! symbols. A double's digits are the shortest that round-trip (Rust's), and
//! the JDK's are `FloatingDecimal`'s older ones, which for some integral
//! doubles past 2^53 carry one more digit (`1e23` formats as
//! `99,999,999,999,999,990,000,000` in Java, `100,000,000,000,000,000,000,000`
//! here).

use super::date_locales::{LocaleData, STRINGS};
use super::DateLocale;
use crate::java_character;

/// What `DecimalFormat.parse` returns: `Long` or `Double`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JavaNumber {
    Long(i64),
    Double(f64),
}

impl JavaNumber {
    /// `Number.intValue()`: a `Long`'s low 32 bits, a `Double` cast (NaN is
    /// 0, saturating).
    pub fn int_value(self) -> i32 {
        match self {
            JavaNumber::Long(v) => v as i32,
            JavaNumber::Double(d) => d as i32,
        }
    }

    /// `Number.longValue()`.
    pub fn long_value(self) -> i64 {
        match self {
            JavaNumber::Long(v) => v,
            JavaNumber::Double(d) => d as i64,
        }
    }

    /// `Number.floatValue()`: rounded to nearest.
    pub fn float_value(self) -> f32 {
        match self {
            JavaNumber::Long(v) => v as f32,
            JavaNumber::Double(d) => d as f32,
        }
    }

    /// `Number.doubleValue()`.
    pub fn double_value(self) -> f64 {
        match self {
            JavaNumber::Long(v) => v as f64,
            JavaNumber::Double(d) => d,
        }
    }
}

/// `NumberFormat.getNumberInstance(locale)` or `getIntegerInstance(locale)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecimalFormat {
    tag: &'static str,
    integer: bool,
    decimal: u16,
    grouping: u16,
    zero: u32,
    grouping_size: usize,
    minus: Vec<u16>,
    nan: Vec<u16>,
    infinity: Vec<u16>,
    exponent: Vec<u16>,
}

impl DecimalFormat {
    fn of(locale: DateLocale, integer: bool) -> Self {
        let d: &LocaleData = locale.data();
        let units = |i: u16| STRINGS[usize::from(i)].encode_utf16().collect::<Vec<u16>>();
        // The table's separators are BMP `char`s, as Java's are.
        let unit = |c: char| u16::try_from(u32::from(c)).unwrap_or(0);
        DecimalFormat {
            tag: locale.tag(),
            integer,
            decimal: unit(d.decimal),
            grouping: unit(d.grouping),
            zero: u32::from(d.zero),
            grouping_size: usize::from(d.grouping_size),
            minus: units(d.minus),
            nan: units(d.nan),
            infinity: units(d.infinity),
            exponent: units(d.exponent),
        }
    }

    /// `NumberFormat.getNumberInstance(locale)`.
    pub fn number_instance(locale: DateLocale) -> Self {
        Self::of(locale, false)
    }

    /// `NumberFormat.getIntegerInstance(locale)`.
    pub fn integer_instance(locale: DateLocale) -> Self {
        Self::of(locale, true)
    }

    /// The locale's `toLanguageTag()`.
    pub fn locale_tag(&self) -> &'static str {
        self.tag
    }

    /// `isParseIntegerOnly()`: the integer instance.
    pub fn is_integer_instance(&self) -> bool {
        self.integer
    }

    /// `NumberFormat.parse(String)`: the number, or `None` where Java throws
    /// `ParseException` (nothing parsed). Text after the number is ignored.
    pub fn parse(&self, text: &str) -> Option<JavaNumber> {
        let t: Vec<u16> = text.encode_utf16().collect();
        self.parse_at(&t, 0).map(|(n, _)| n)
    }

    /// `parse(text, new ParsePosition(pos))`: the number and the index the
    /// parse ended at, or `None` (the position's error index set).
    pub fn parse_at(&self, t: &[u16], pos: usize) -> Option<(JavaNumber, usize)> {
        let starts =
            |i: usize, s: &[u16]| !s.is_empty() && t.get(i..).is_some_and(|r| r.starts_with(s));
        if starts(pos, &self.nan) {
            return Some((JavaNumber::Double(f64::NAN), pos + self.nan.len()));
        }
        let mut p = pos;
        let negative = starts(p, &self.minus);
        if negative {
            p += self.minus.len();
        }
        if starts(p, &self.infinity) {
            let inf = if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            return Some((JavaNumber::Double(inf), p + self.infinity.len()));
        }
        let digit = |c: u16| -> Option<u8> {
            if (0xD800..=0xDFFF).contains(&c) {
                return None;
            }
            java_character::decimal_digit_value(u32::from(c)).and_then(|d| u8::try_from(d).ok())
        };
        // The significant digits (no leading zero) and where the decimal
        // point falls among them.
        let mut digits: Vec<u8> = Vec::new();
        let mut decimal_at: i64 = 0;
        let mut saw_digit = false;
        let mut saw_decimal = false;
        let mut backup: Option<usize> = None;
        let mut exponent: i64 = 0;
        while let Some(&c) = t.get(p) {
            if let Some(d) = digit(c) {
                saw_digit = true;
                backup = None;
                if d == 0 && digits.is_empty() {
                    if saw_decimal {
                        decimal_at -= 1;
                    }
                } else {
                    digits.push(d);
                    if !saw_decimal {
                        decimal_at += 1;
                    }
                }
                p += 1;
            } else if c == self.decimal {
                if self.integer {
                    // JDK 25: the parse ends at the separator, nothing given
                    // back, and after a grouping separator a digit following
                    // it counts as one seen (`,.5` is 0, `.5` fails).
                    saw_digit |=
                        backup.is_some() && t.get(p + 1).copied().and_then(digit).is_some();
                    backup = None;
                    break;
                }
                if saw_decimal {
                    break;
                }
                saw_decimal = true;
                p += 1;
            } else if c == self.grouping {
                if saw_decimal {
                    break;
                }
                backup = Some(p);
                p += 1;
            } else {
                if starts(p, &self.exponent) {
                    let mut q = p + self.exponent.len();
                    let exp_negative = starts(q, &self.minus);
                    if exp_negative {
                        q += self.minus.len();
                    }
                    let from = q;
                    let mut e: i64 = 0;
                    while let Some(d) = t.get(q).copied().and_then(digit) {
                        e = e.saturating_mul(10).saturating_add(i64::from(d));
                        q += 1;
                    }
                    if q > from {
                        exponent = if exp_negative { -e } else { e };
                        p = q;
                    }
                }
                break;
            }
        }
        if let Some(b) = backup {
            p = b;
        }
        if !saw_digit {
            return None;
        }
        Some((self.value(negative, digits, decimal_at, exponent), p))
    }

    /// The `Long` or `Double` of the digits read.
    fn value(&self, negative: bool, mut digits: Vec<u8>, decimal_at: i64, exp: i64) -> JavaNumber {
        while digits.last() == Some(&0) {
            digits.pop();
        }
        if digits.is_empty() {
            return if negative && !self.integer {
                JavaNumber::Double(-0.0)
            } else {
                JavaNumber::Long(0)
            };
        }
        let decimal_at = decimal_at
            .saturating_add(exp)
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX));
        let count = i64::try_from(digits.len()).unwrap_or(i64::MAX);
        if decimal_at >= count && decimal_at <= 19 {
            let mut v: i128 = 0;
            for i in 0..decimal_at {
                let d = usize::try_from(i)
                    .ok()
                    .and_then(|i| digits.get(i))
                    .copied()
                    .unwrap_or(0);
                v = v * 10 + i128::from(d);
            }
            if let Ok(long) = i64::try_from(if negative { -v } else { v }) {
                return JavaNumber::Long(long);
            }
        }
        let text: String = digits.iter().map(|&d| char::from(b'0' + d)).collect();
        let double: f64 = format!(".{text}E{decimal_at}").parse().unwrap_or(0.0);
        JavaNumber::Double(if negative { -double } else { double })
    }

    /// `format(long)`.
    pub fn format_long(&self, v: i64) -> String {
        let mut out = String::new();
        if v < 0 {
            out.push_str(&String::from_utf16_lossy(&self.minus));
        }
        self.push_integer(&mut out, &v.unsigned_abs().to_string());
        out
    }

    /// `format(double)`.
    pub fn format_double(&self, v: f64) -> String {
        if v.is_nan() {
            return String::from_utf16_lossy(&self.nan);
        }
        let mut out = String::new();
        if v.is_sign_negative() {
            out.push_str(&String::from_utf16_lossy(&self.minus));
        }
        let abs = v.abs();
        if abs.is_infinite() {
            out.push_str(&String::from_utf16_lossy(&self.infinity));
            return out;
        }
        let max_fraction: usize = if self.integer { 0 } else { 3 };
        let (int_part, fraction) = decimal_digits(abs, max_fraction);
        self.push_integer(&mut out, &int_part);
        let fraction = fraction.trim_end_matches('0');
        if !fraction.is_empty() {
            out.push(char::from_u32(u32::from(self.decimal)).unwrap_or('.'));
            for b in fraction.bytes() {
                out.push(self.digit_char(b));
            }
        }
        out
    }

    fn digit_char(&self, ascii: u8) -> char {
        char::from_u32(self.zero + u32::from(ascii - b'0')).unwrap_or(char::from(ascii))
    }

    /// The integer digits (ASCII, no leading zero but a lone `0`), grouped.
    fn push_integer(&self, out: &mut String, digits: &str) {
        let digits = digits.trim_start_matches('0');
        let digits = if digits.is_empty() { "0" } else { digits };
        let grouping = char::from_u32(u32::from(self.grouping)).unwrap_or(',');
        let n = digits.len();
        for (i, b) in digits.bytes().enumerate() {
            if i > 0 && self.grouping_size > 0 && (n - i) % self.grouping_size == 0 {
                out.push(grouping);
            }
            out.push(self.digit_char(b));
        }
    }
}

/// A finite non-negative double as ASCII integer digits and at most
/// `max_fraction` fraction digits, rounded half-even: the shortest
/// round-tripping digits when they need no rounding, else the exact value
/// rounded (where the shortest digits would round the same way unless they
/// sit on the tie, which the exact value then breaks -- as Java's
/// `DigitList` does).
fn decimal_digits(abs: f64, max_fraction: usize) -> (String, String) {
    let sci = format!("{abs:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i64 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let len = i64::try_from(digits.len()).unwrap_or(i64::MAX);
    let fraction_needed = (len - 1 - exp).max(0);
    if fraction_needed <= i64::try_from(max_fraction).unwrap_or(0) {
        // The digits as they are: `d.ddd` times ten to `exp`.
        let point = exp + 1;
        if point <= 0 {
            let zeros = usize::try_from(-point).unwrap_or(0);
            return (String::new(), format!("{}{digits}", "0".repeat(zeros)));
        }
        let point = usize::try_from(point).unwrap_or(0);
        if point >= digits.len() {
            return (
                format!("{digits}{}", "0".repeat(point - digits.len())),
                String::new(),
            );
        }
        return (digits[..point].to_string(), digits[point..].to_string());
    }
    let fixed = format!("{abs:.max_fraction$}");
    match fixed.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (fixed, String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn locale(tag: &str) -> DateLocale {
        DateLocale::for_language_tag(tag).expect(tag)
    }

    fn num(tag: &str) -> DecimalFormat {
        DecimalFormat::number_instance(locale(tag))
    }

    fn int(tag: &str) -> DecimalFormat {
        DecimalFormat::integer_instance(locale(tag))
    }

    /// `parse(text, new ParsePosition(0))` as the probes printed it.
    fn probe(f: &DecimalFormat, text: &str) -> String {
        let t: Vec<u16> = text.encode_utf16().collect();
        match f.parse_at(&t, 0) {
            None => "null".to_string(),
            Some((JavaNumber::Long(v), end)) => format!("Long:{v}@{end}"),
            Some((JavaNumber::Double(d), end)) => format!("Double:{d:?}@{end}"),
        }
    }

    #[test]
    fn parses_as_jdk_25() {
        // (instance, text, JDK 25's `n.getClass().getSimpleName() + ":" + n` and index)
        let us = [
            (num("en-US"), "1,234", "Long:1234@5"),
            (num("en-US"), "1,,2", "Long:12@4"),
            (num("en-US"), "1,", "Long:1@1"),
            (num("en-US"), ",1", "Long:1@2"),
            (num("en-US"), ",", "null"),
            (num("en-US"), "1.", "Long:1@2"),
            (num("en-US"), "-.5", "Double:-0.5@3"),
            (num("en-US"), "1.2.3", "Double:1.2@3"),
            (num("en-US"), "1,234.5,6", "Double:1234.5@7"),
            (num("en-US"), "-0", "Double:-0.0@2"),
            (num("en-US"), "0.0", "Long:0@3"),
            (num("en-US"), "-NaN", "null"),
            (num("en-US"), "NaN", "Double:NaN@3"),
            (num("en-US"), "-∞", "Double:-inf@2"),
            (num("en-US"), "1E-3", "Double:0.001@4"),
            (num("en-US"), "1E", "Long:1@1"),
            (num("en-US"), "1e3", "Long:1@1"),
            (num("en-US"), "1E+3", "Long:1@1"),
            (
                num("en-US"),
                "9223372036854775808",
                "Double:9.223372036854776e18@19",
            ),
            (
                num("en-US"),
                "-9223372036854775808",
                "Long:-9223372036854775808@20",
            ),
            (num("en-US"), "1E19", "Double:1e19@4"),
            (num("en-US"), "12E17", "Long:1200000000000000000@5"),
            (num("en-US"), "1E-400", "Double:0.0@6"),
            (num("en-US"), "-1E-400", "Double:-0.0@7"),
            (num("en-US"), "1E400", "Double:inf@5"),
            (num("en-US"), "١٢٣", "Long:123@3"),
            (num("en-US"), "1,.5", "Double:1.5@4"),
            (num("en-US"), "1,,.", "Long:1@2"),
            (num("en-US"), "1,.,", "Long:1@1"),
            (num("en-US"), "1,E3", "Long:1000@1"),
            (num("en-US"), "1,.E3", "Long:1000@1"),
            (num("en-US"), "1E3,", "Long:1000@3"),
            (num("en-US"), "-,0", "Double:-0.0@3"),
            (num("en-US"), "0.000000000000000000001E21", "Long:1@26"),
            (num("en-US"), "1E99999999999999999999", "Double:inf@22"),
            (num("en-US"), "1E-99999999999999999999", "Double:0.0@23"),
            (int("en-US"), "1.5", "Long:1@1"),
            (int("en-US"), ".5", "null"),
            (int("en-US"), "-.5", "null"),
            (int("en-US"), "-", "null"),
            (int("en-US"), ",,.5", "Long:0@2"),
            (int("en-US"), ",.,", "null"),
            (num("en-US"), ".5", "Double:0.5@2"),
            (num("en-US"), "0.5E1", "Long:5@5"),
            (num("en-US"), "1.5E2", "Long:150@5"),
            (num("en-US"), "-00.000", "Double:-0.0@7"),
            (num("en-US"), "99999999999999999999", "Double:1e20@20"),
            (num("en-US"), "1E-", "Long:1@1"),
            (num("en-US"), "0E5", "Long:0@3"),
            (int("en-US"), "-0", "Long:0@2"),
            (int("en-US"), "1E-3", "Double:0.001@4"),
            (int("en-US"), "1.5E2", "Long:1@1"),
            (int("en-US"), "1,000,000.123456789", "Long:1000000@9"),
            // JDK 25 only (JDK 21: `Long:1@1`, `Long:1@2`, `null`)
            (int("en-US"), "1,.5", "Long:1@2"),
            (int("en-US"), "1,,.", "Long:1@3"),
            (int("en-US"), ",.5", "Long:0@1"),
            (int("en-US"), "-,.5", "Long:0@2"),
            (int("en-US"), ",.5E3", "Long:0@1"),
            (int("en-US"), ",.", "null"),
            (int("en-US"), ",.x", "null"),
            (num("de-DE"), "1.000,5", "Double:1000.5@7"),
            (num("fr-FR"), "1\u{202f}000", "Long:1000@5"),
            (num("fr-FR"), "1\u{a0}000", "Long:1@1"),
            (num("sv-SE"), "\u{2212}3", "Long:-3@2"),
            (num("sv-SE"), "-3", "null"),
            (num("de-CH"), "1\u{2019}000", "Long:1000@5"),
            (num("fa"), "\u{200e}\u{2212}3", "Long:-3@3"),
            (num("fa"), "NaN", "null"),
            (num("fa"), "-3", "null"),
            // `tok`: grouping by two (JDK 21's data is `#,##0.###`)
            (num("tok"), "1,5", "Double:1.5@3"),
            (num("tok"), "1\u{a0}000", "Long:1000@5"),
        ];
        for (f, text, want) in us {
            assert_eq!(probe(&f, text), want, "{} {text:?}", f.locale_tag());
        }
        assert_eq!(num("en-US").parse("1x"), Some(JavaNumber::Long(1)));
        assert_eq!(num("en-US").parse(""), None);
        assert!(!num("en-US").is_integer_instance() && int("en-US").is_integer_instance());
    }

    #[test]
    fn formats_as_jdk_25() {
        let cases: &[(&DecimalFormat, f64, &str)] = &[
            (&num("en-US"), 0.0, "0"),
            (&num("en-US"), -0.0, "-0"),
            (&num("en-US"), 1.5, "1.5"),
            (&num("en-US"), 0.0005, "0.001"),
            (&num("en-US"), 0.0015, "0.002"),
            (&num("en-US"), 0.0025, "0.003"),
            (&num("en-US"), 0.0625, "0.062"),
            (&num("en-US"), 0.125, "0.125"),
            (&num("en-US"), 1.0005, "1"),
            (&num("en-US"), 1234567.891, "1,234,567.891"),
            (&num("en-US"), 1e22, "10,000,000,000,000,000,000,000"),
            (
                &num("en-US"),
                123456789012345678.0,
                "123,456,789,012,345,680",
            ),
            (&num("en-US"), 1e-7, "0"),
            (&num("en-US"), -0.0001, "-0"),
            (&num("en-US"), f64::NAN, "NaN"),
            (&num("en-US"), f64::INFINITY, "∞"),
            (&num("en-US"), f64::NEG_INFINITY, "-∞"),
            (&num("en-US"), -1.25, "-1.25"),
            (&num("en-US"), 0.1235, "0.123"),
            (&num("en-US"), 0.12350000001, "0.124"),
            (&num("en-US"), f64::from(1.1f32), "1.1"),
            (&num("en-US"), 2.675, "2.675"),
            (&int("en-US"), 1.5, "2"),
            (&int("en-US"), 2.5, "2"),
            (&int("en-US"), 0.5, "0"),
            (&int("en-US"), -0.4, "-0"),
            (&int("en-US"), -1234.5678, "-1,235"),
            (&num("fa"), -1234.5678, "\u{200e}\u{2212}۱٬۲۳۴٫۵۶۸"),
            (&num("de-CH"), 1e7, "10\u{2019}000\u{2019}000"),
            (&num("fr-FR"), -1234.5678, "-1\u{202f}234,568"),
            (&num("tok"), 1e7, "10\u{a0}00\u{a0}00\u{a0}00"),
        ];
        for (f, v, want) in cases {
            assert_eq!(f.format_double(*v), *want, "{} {v}", f.locale_tag());
        }
        assert_eq!(
            num("en-US").format_long(i64::MIN),
            "-9,223,372,036,854,775,808"
        );
        assert_eq!(num("en-US").format_long(0), "0");
        assert_eq!(num("ar-EG").format_long(-1234), "\u{61c}-١٬٢٣٤");
        assert_eq!(
            num("tok").format_long(-1234567),
            "-1\u{a0}23\u{a0}45\u{a0}67"
        );
    }

    #[test]
    fn number_values_are_java_conversions() {
        let big = JavaNumber::Long(1 << 40 | 5);
        assert_eq!(big.int_value(), 5);
        assert_eq!(big.long_value(), 1 << 40 | 5);
        assert_eq!(JavaNumber::Double(1e300).int_value(), i32::MAX);
        assert_eq!(JavaNumber::Double(f64::NAN).long_value(), 0);
        assert_eq!(JavaNumber::Double(1e300).float_value(), f32::INFINITY);
        assert_eq!(JavaNumber::Long(16_777_217).float_value(), 16_777_216.0);
        assert_eq!(JavaNumber::Long(3).double_value(), 3.0);
        assert_eq!(JavaNumber::Double(0.5).double_value(), 0.5);
    }
}
