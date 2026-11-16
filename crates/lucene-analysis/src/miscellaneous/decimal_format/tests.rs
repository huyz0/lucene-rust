#![cfg(test)]
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
