#![cfg(test)]
use super::*;
use crate::api::test_support::{call, i, s, t, text, vm_call};

fn st(class: &str, name: &str, args: &[Value]) -> String {
    match static_lookup(class, name, args.len()) {
        Some(f) => text(vm_call(f, args)),
        None => format!("no static {class}.{name}/{}", args.len()),
    }
}

fn l(x: i64) -> Value {
    Value::Long(x)
}

fn d(x: f64) -> Value {
    Value::Double(x)
}

fn c(x: char) -> Value {
    Value::Char(x as u16)
}

/// `(class, name, args) -> expected`, each a line of Java's documented
/// behaviour.
#[test]
fn integer_statics() {
    let cases: Vec<(&str, Vec<Value>, &str)> = vec![
        ("parseInt", vec![s("-42")], "-42"),
        ("parseInt", vec![s("ff"), i(16)], "255"),
        ("valueOf", vec![s("12")], "12"),
        ("valueOf", vec![i(12)], "12"),
        ("valueOf", vec![s("101"), i(2)], "5"),
        ("toString", vec![i(-7)], "-7"),
        ("toString", vec![i(255), i(16)], "ff"),
        ("toString", vec![i(-255), i(16)], "-ff"),
        ("toHexString", vec![i(-1)], "ffffffff"),
        ("toBinaryString", vec![i(5)], "101"),
        ("toOctalString", vec![i(8)], "10"),
        ("compare", vec![i(1), i(2)], "-1"),
        ("compare", vec![i(2), i(2)], "0"),
        ("max", vec![i(3), i(9)], "9"),
        ("min", vec![i(3), i(9)], "3"),
        ("sum", vec![i(i32::MAX), i(1)], "-2147483648"),
        ("signum", vec![i(-9)], "-1"),
        ("bitCount", vec![i(255)], "8"),
        ("reverse", vec![i(1)], "-2147483648"),
        ("reverseBytes", vec![i(1)], "16777216"),
        ("highestOneBit", vec![i(100)], "64"),
        ("highestOneBit", vec![i(0)], "0"),
        ("lowestOneBit", vec![i(12)], "4"),
        ("numberOfLeadingZeros", vec![i(1)], "31"),
        ("numberOfTrailingZeros", vec![i(8)], "3"),
        ("rotateLeft", vec![i(1), i(33)], "2"),
        ("rotateRight", vec![i(1), i(1)], "-2147483648"),
        ("hashCode", vec![i(77)], "77"),
        ("toUnsignedLong", vec![i(-1)], "4294967295"),
        ("toUnsignedString", vec![i(-1)], "4294967295"),
        ("compareUnsigned", vec![i(-1), i(1)], "1"),
        ("divideUnsigned", vec![i(-1), i(2)], "2147483647"),
        ("remainderUnsigned", vec![i(-1), i(2)], "1"),
        ("parseUnsignedInt", vec![s("4294967295")], "-1"),
        ("decode", vec![s("0x1F")], "31"),
        ("decode", vec![s("-010")], "-8"),
        ("decode", vec![s("#ff")], "255"),
    ];
    for (name, args, want) in cases {
        assert_eq!(st("Integer", name, &args), want, "Integer.{name}");
    }
    assert!(st("Integer", "parseInt", &[s("x")]).contains("NumberFormatException"));
    assert!(st("Integer", "parseInt", &[s("2147483648")]).contains("NumberFormatException"));
    assert!(
        st("Integer", "parseInt", &[Value::Null]).contains("NumberFormatException")
            || st("Integer", "parseInt", &[Value::Null]).contains("NullPointer")
    );
    assert!(st("Integer", "divideUnsigned", &[i(1), i(0)]).contains("Arithmetic"));
}

#[test]
fn long_short_byte_statics() {
    let cases: Vec<(&str, &str, Vec<Value>, &str)> = vec![
        ("Long", "parseLong", vec![s("9000000000")], "9000000000"),
        ("Long", "parseLong", vec![s("-ff"), i(16)], "-255"),
        ("Long", "valueOf", vec![l(5)], "5"),
        ("Long", "toString", vec![l(-5)], "-5"),
        ("Long", "toString", vec![l(255), i(2)], "11111111"),
        ("Long", "toHexString", vec![l(-1)], "ffffffffffffffff"),
        ("Long", "toBinaryString", vec![l(6)], "110"),
        ("Long", "compare", vec![l(3), l(1)], "1"),
        ("Long", "max", vec![l(3), l(1)], "3"),
        ("Long", "min", vec![l(3), l(1)], "1"),
        (
            "Long",
            "sum",
            vec![l(i64::MAX), l(1)],
            "-9223372036854775808",
        ),
        ("Long", "signum", vec![l(-3)], "-1"),
        ("Long", "bitCount", vec![l(-1)], "64"),
        ("Long", "numberOfLeadingZeros", vec![l(1)], "63"),
        ("Long", "numberOfTrailingZeros", vec![l(16)], "4"),
        ("Long", "hashCode", vec![l(1 << 32)], "1"),
        ("Long", "reverse", vec![l(1)], "-9223372036854775808"),
        ("Short", "parseShort", vec![s("-300")], "-300"),
        ("Short", "valueOf", vec![s("7")], "7"),
        ("Short", "toString", vec![Value::Short(9)], "9"),
        (
            "Short",
            "compare",
            vec![Value::Short(1), Value::Short(5)],
            "-4",
        ),
        ("Byte", "parseByte", vec![s("127")], "127"),
        ("Byte", "valueOf", vec![s("-128")], "-128"),
        ("Byte", "toString", vec![Value::Byte(-3)], "-3"),
        (
            "Byte",
            "compare",
            vec![Value::Byte(1), Value::Byte(5)],
            "-4",
        ),
    ];
    for (class, name, args, want) in cases {
        assert_eq!(st(class, name, &args), want, "{class}.{name}");
    }
    assert!(st("Short", "parseShort", &[s("40000")]).contains("NumberFormatException"));
    assert!(st("Byte", "parseByte", &[s("128")]).contains("NumberFormatException"));
    assert!(st("Long", "parseLong", &[s("")]).contains("NumberFormatException"));
}

#[test]
fn floating_point_statics() {
    let cases: Vec<(&str, &str, Vec<Value>, &str)> = vec![
        ("Double", "parseDouble", vec![s("2.5")], "2.5"),
        ("Double", "valueOf", vec![s("-1e3")], "-1000.0"),
        ("Float", "parseFloat", vec![s("1.5")], "1.5"),
        ("Float", "valueOf", vec![s("2")], "2.0"),
        ("Double", "toString", vec![d(0.1)], "0.1"),
        ("Float", "toString", vec![Value::Float(0.5)], "0.5"),
        ("Double", "compare", vec![d(1.0), d(2.0)], "-1"),
        ("Double", "compare", vec![d(f64::NAN), d(1.0)], "1"),
        (
            "Float",
            "compare",
            vec![Value::Float(2.0), Value::Float(2.0)],
            "0",
        ),
        ("Double", "isNaN", vec![d(f64::NAN)], "true"),
        ("Double", "isInfinite", vec![d(f64::INFINITY)], "true"),
        ("Double", "isFinite", vec![d(1.0)], "true"),
        ("Double", "isFinite", vec![d(f64::NAN)], "false"),
        ("Double", "max", vec![d(1.0), d(2.0)], "2.0"),
        ("Double", "max", vec![d(f64::NAN), d(2.0)], "NaN"),
        ("Double", "min", vec![d(-0.0), d(0.0)], "-0.0"),
        ("Double", "sum", vec![d(0.5), d(0.25)], "0.75"),
        (
            "Double",
            "doubleToLongBits",
            vec![d(1.0)],
            "4607182418800017408",
        ),
        (
            "Double",
            "doubleToRawLongBits",
            vec![d(1.0)],
            "4607182418800017408",
        ),
        (
            "Double",
            "longBitsToDouble",
            vec![l(4607182418800017408)],
            "1.0",
        ),
        ("Double", "hashCode", vec![d(1.0)], "1072693248"),
        (
            "Float",
            "floatToIntBits",
            vec![Value::Float(1.0)],
            "1065353216",
        ),
        ("Float", "intBitsToFloat", vec![i(1065353216)], "1.0"),
        (
            "Float",
            "max",
            vec![Value::Float(1.0), Value::Float(3.0)],
            "3.0",
        ),
        (
            "Float",
            "min",
            vec![Value::Float(1.0), Value::Float(3.0)],
            "1.0",
        ),
    ];
    for (class, name, args, want) in cases {
        assert_eq!(st(class, name, &args), want, "{class}.{name}");
    }
    assert!(st("Double", "parseDouble", &[s("abc")]).contains("NumberFormatException"));
}

#[test]
fn boolean_and_character_statics() {
    let b = Value::Bool;
    let cases: Vec<(&str, &str, Vec<Value>, &str)> = vec![
        ("Boolean", "parseBoolean", vec![s("TRUE")], "true"),
        ("Boolean", "parseBoolean", vec![s("yes")], "false"),
        ("Boolean", "valueOf", vec![b(true)], "true"),
        ("Boolean", "toString", vec![b(false)], "false"),
        ("Boolean", "compare", vec![b(true), b(false)], "1"),
        ("Boolean", "logicalAnd", vec![b(true), b(false)], "false"),
        ("Boolean", "logicalOr", vec![b(true), b(false)], "true"),
        ("Boolean", "logicalXor", vec![b(true), b(true)], "false"),
        ("Boolean", "hashCode", vec![b(true)], "1231"),
        ("Boolean", "hashCode", vec![b(false)], "1237"),
        ("Character", "isDigit", vec![c('7')], "true"),
        ("Character", "isLetter", vec![c('x')], "true"),
        ("Character", "isLetterOrDigit", vec![c('_')], "false"),
        ("Character", "isAlphabetic", vec![c('z')], "true"),
        ("Character", "isWhitespace", vec![c('\t')], "true"),
        ("Character", "isSpaceChar", vec![c(' ')], "true"),
        ("Character", "isUpperCase", vec![c('A')], "true"),
        ("Character", "isLowerCase", vec![c('A')], "false"),
        ("Character", "toUpperCase", vec![c('a')], "A"),
        ("Character", "toLowerCase", vec![c('A')], "a"),
        ("Character", "toString", vec![c('q')], "q"),
        ("Character", "valueOf", vec![c('q')], "q"),
        ("Character", "compare", vec![c('a'), c('c')], "-2"),
        ("Character", "getNumericValue", vec![c('9')], "9"),
        ("Character", "getNumericValue", vec![c('a')], "10"),
        ("Character", "digit", vec![c('f'), i(16)], "15"),
        ("Character", "digit", vec![c('g'), i(16)], "-1"),
        ("Character", "forDigit", vec![i(11), i(16)], "b"),
        ("Character", "hashCode", vec![c('a')], "97"),
        ("Character", "isISOControl", vec![c('\n')], "true"),
        ("Character", "charCount", vec![i(0x1F600)], "2"),
        ("Character", "charCount", vec![i(65)], "1"),
    ];
    for (class, name, args, want) in cases {
        assert_eq!(st(class, name, &args), want, "{class}.{name}");
    }
}

#[test]
fn boxed_receivers() {
    assert_eq!(t(&i(5), "intValue", &[]), "5");
    assert_eq!(t(&i(5), "longValue", &[]), "5");
    assert_eq!(t(&i(5), "doubleValue", &[]), "5.0");
    assert_eq!(t(&i(5), "floatValue", &[]), "5.0");
    assert_eq!(t(&i(70000), "shortValue", &[]), "4464");
    assert_eq!(t(&i(300), "byteValue", &[]), "44");
    assert_eq!(t(&Value::Bool(true), "booleanValue", &[]), "true");
    assert_eq!(t(&c('x'), "charValue", &[]), "x");
    assert_eq!(t(&i(5), "toString", &[]), "5");
    assert_eq!(t(&l(5), "hashCode", &[]), "5");
    assert_eq!(t(&i(5), "equals", &[i(5)]), "true");
    assert_eq!(t(&i(5), "equals", &[l(5)]), "false");
    assert_eq!(t(&i(5), "compareTo", &[i(9)]), "-1");
    assert_eq!(t(&d(f64::NAN), "isNaN", &[]), "true");
    assert_eq!(t(&d(f64::NEG_INFINITY), "isInfinite", &[]), "true");
    assert!(call(&Value::Bool(true), "intValue", &[]).is_err());
}

#[test]
fn radix_and_decode_helpers() {
    assert_eq!(to_string_radix(0, 10), "0");
    assert_eq!(to_string_radix(-35, 36), "-z");
    assert_eq!(to_string_radix(10, 99), "10");
    assert_eq!(decode("0X10", -100, 100).unwrap(), 16);
    assert_eq!(decode("+7", -100, 100).unwrap(), 7);
    assert!(decode("", -100, 100).is_err());
    assert!(decode("300", -100, 100).is_err());
    assert!(format!(
        "{:?}",
        parse_int("12", 99, -100, 100).unwrap_err().message()
    )
    .contains("radix 99 greater than Character.MAX_RADIX"));
    assert!(
        format!("{:?}", parse_int("12", 1, -100, 100).unwrap_err().message())
            .contains("radix 1 less than Character.MIN_RADIX")
    );
    assert!(st("Integer", "parseInt", &[s("12"), i(99)]).contains("NumberFormatException"));
    assert_eq!(parse_int("-0", 10, -100, 100).unwrap(), 0);
}
