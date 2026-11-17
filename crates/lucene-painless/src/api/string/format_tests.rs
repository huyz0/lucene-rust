#![cfg(test)]
//! `String.format` conversions and the `StringBuilder` members, with the
//! results `java.util.Formatter` and `java.lang.StringBuilder` give.
use super::*;
use crate::api::test_support::{call, i, s, t, text, vm_call};

fn fmt(f: &str, args: &[Value]) -> String {
    text(format(f, args).map(Value::string))
}

fn d(x: f64) -> Value {
    Value::Double(x)
}

fn builder(init: &str) -> Value {
    vm_call(builder_constructor(1).unwrap(), &[s(init)]).unwrap()
}

#[test]
fn string_and_flag_conversions() {
    assert_eq!(fmt("%.2s", &[s("abcdef")]), "ab");
    assert_eq!(fmt("%S", &[s("abc")]), "ABC");
    assert_eq!(fmt("%b", &[Value::Null]), "false");
    assert_eq!(fmt("%b", &[Value::Bool(true)]), "true");
    assert_eq!(fmt("%B", &[s("x")]), "TRUE");
    assert_eq!(fmt("%c", &[Value::Char(u16::from(b'z'))]), "z");
    assert_eq!(fmt("%c", &[i(65)]), "A");
    assert_eq!(fmt("%10s|", &[s("hi")]), format!("{}hi|", " ".repeat(8)));
    assert_eq!(fmt("%-10s|", &[s("hi")]), format!("hi{}|", " ".repeat(8)));
    assert_eq!(fmt("%-6d|", &[i(42)]), "42    |");
    assert_eq!(fmt("%05d", &[i(42)]), "00042");
    assert_eq!(fmt("%,d", &[i(1_234_567)]), "1,234,567");
    assert_eq!(fmt("%+d", &[i(5)]), "+5");
    assert_eq!(fmt("%d", &[i(-5)]), "-5");
    assert_eq!(fmt("%,d", &[Value::Long(-1_234)]), "-1,234");
    assert_eq!(fmt("%x", &[i(255)]), "ff");
    assert_eq!(fmt("%X", &[i(255)]), "FF");
    assert_eq!(fmt("%x", &[i(-1)]), "ffffffff");
    assert_eq!(fmt("%x", &[Value::Long(-1)]), "ffffffffffffffff");
    assert_eq!(fmt("%o", &[i(8)]), "10");
    assert_eq!(fmt("%o", &[Value::Long(64)]), "100");
    assert_eq!(fmt("%2$s %1$s", &[s("a"), s("b")]), "b a");
    assert_eq!(fmt("a%nb", &[]), "a\nb");
    assert_eq!(fmt("%s %s", &[s("a"), Value::Null]), "a null");
}

#[test]
fn floating_point_conversions() {
    assert_eq!(fmt("%.2f", &[d(-0.001)]), "-0.00");
    assert_eq!(fmt("%.0f", &[d(-0.4)]), "-0");
    assert_eq!(fmt("%.2f", &[d(-0.0)]), "-0.00");
    assert_eq!(fmt("%+.1f", &[d(2.55)]), "+2.6");
    assert_eq!(fmt("%,.2f", &[d(1_234_567.891)]), "1,234,567.89");
    assert_eq!(fmt("%,.2f", &[d(-1_234.5)]), "-1,234.50");
    assert_eq!(fmt("%06.2f", &[d(-3.24159)]), "-03.24");
    assert_eq!(fmt("%08.3f", &[d(3.24159)]), "0003.242");
    assert_eq!(fmt("%e", &[d(12345.678)]), "1.234568e+04");
    assert_eq!(fmt("%E", &[d(12345.678)]), "1.234568E+04");
    assert_eq!(fmt("%e", &[d(0.00012)]), "1.200000e-04");
    assert_eq!(fmt("%.2e", &[d(-5.0)]), "-5.00e+00");
    assert_eq!(fmt("%.1f", &[d(9.99)]), "10.0");
    assert_eq!(fmt("%.6f", &[d(1.5e-5)]), "0.000015");
    assert_eq!(fmt("%.2f", &[d(1e10)]), "10000000000.00");
    assert_eq!(fmt("%.1f", &[Value::Float(0.25)]), "0.3");
    assert_eq!(format_fixed(f64::NAN, 2), "NaN");
    assert_eq!(format_fixed(f64::INFINITY, 2), "Infinity");
    assert_eq!(format_fixed(f64::NEG_INFINITY, 2), "-Infinity");
}

#[test]
fn bad_formats_throw_javas_exceptions() {
    assert_eq!(
        fmt("%", &[]),
        "java.util.UnknownFormatConversionException: Conversion = '%'"
    );
    assert_eq!(
        fmt("%q", &[s("s")]),
        "java.util.UnknownFormatConversionException: Conversion = 'q'"
    );
    assert_eq!(
        fmt("%d", &[s("s")]),
        "java.util.IllegalFormatConversionException: d != java.lang.String"
    );
    assert_eq!(
        fmt("%d", &[]),
        "java.util.MissingFormatArgumentException: Format specifier '%d'"
    );
    for conv in ["c", "x", "o", "f"] {
        assert!(
            fmt(&format!("%{conv}"), &[s("s")]).contains("IllegalFormatConversionException"),
            "{conv}"
        );
    }
}

#[test]
fn builder_edits() {
    let b = builder("hello");
    assert_eq!(t(&b, "append", &[s("!")]), "hello!");
    assert_eq!(t(&b, "length", &[]), "6");
    assert_eq!(t(&builder(""), "append", &[s("xyzw"), i(1), i(3)]), "yz");
    assert_eq!(
        t(&builder("A"), "appendCodePoint", &[i(0x1F600)]),
        "A\u{1F600}"
    );
    assert_eq!(t(&builder("hello"), "reverse", &[]), "olleh");
    assert_eq!(t(&builder("hello"), "insert", &[i(0), s("ab")]), "abhello");
    assert_eq!(t(&builder("hello"), "insert", &[i(5), s("!")]), "hello!");
    assert_eq!(t(&builder("hello"), "delete", &[i(1), i(3)]), "hlo");
    assert_eq!(t(&builder("hello"), "delete", &[i(2), i(100)]), "he");
    assert_eq!(t(&builder("hello"), "deleteCharAt", &[i(0)]), "ello");
    assert_eq!(
        t(&builder("hello"), "replace", &[i(1), i(100), s("XY")]),
        "hXY"
    );
    assert_eq!(
        t(&builder("hello"), "replace", &[i(1), i(3), s("XY")]),
        "hXYlo"
    );
    assert_eq!(t(&builder("hello"), "toString", &[]), "hello");
    for (name, args) in [
        ("insert", vec![i(9), s("x")]),
        ("insert", vec![i(-1), s("x")]),
        ("delete", vec![i(3), i(1)]),
        ("delete", vec![i(-1), i(1)]),
        ("deleteCharAt", vec![i(5)]),
        ("deleteCharAt", vec![i(-1)]),
        ("setCharAt", vec![i(9), Value::Char(u16::from(b'x'))]),
    ] {
        assert!(
            t(&builder("hello"), name, &args).contains("StringIndexOutOfBounds"),
            "{name}"
        );
    }
    assert!(t(&builder("hello"), "setCharAt", &[i(0), s("x")]).contains("ClassCast"));
}

#[test]
fn builder_length_chars_and_reads() {
    let b = builder("hello");
    call(&b, "setLength", &[i(2)]).unwrap();
    assert_eq!(t(&b, "toString", &[]), "he");
    call(&b, "setLength", &[i(4)]).unwrap();
    assert_eq!(t(&b, "length", &[]), "4");
    assert_eq!(t(&b, "charAt", &[i(3)]), "\0");
    let c = builder("hello");
    call(&c, "setCharAt", &[i(0), Value::Char(u16::from(b'J'))]).unwrap();
    assert_eq!(t(&c, "toString", &[]), "Jello");
    assert_eq!(t(&c, "indexOf", &[s("l")]), "2");
    assert_eq!(t(&c, "indexOf", &[s("l"), i(3)]), "3");
    assert_eq!(t(&c, "lastIndexOf", &[s("l")]), "3");
    assert!(t(&c, "indexOf", &[i(1)]).contains("ClassCast"));
    assert_eq!(t(&c, "charAt", &[i(1)]), "e");
    assert_eq!(t(&c, "substring", &[i(1), i(3)]), "el");
    assert_eq!(t(&c, "substring", &[i(3)]), "lo");
    assert_eq!(t(&c, "subSequence", &[i(0), i(2)]), "Je");
    assert_eq!(t(&c, "codePointAt", &[i(0)]), "74");
    assert_eq!(t(&builder("hello"), "compareTo", &[builder("help")]), "-4");
    assert_eq!(t(&builder("a"), "compareTo", &[builder("a")]), "0");
    assert!(t(&builder("a"), "compareTo", &[Value::Null]).contains("NullPointer"));
    assert!(t(&builder("a"), "compareTo", &[s("a")]).contains("ClassCast"));
    assert!(call(&builder("a"), "capacity", &[]).is_err());
}

#[test]
fn builder_constructors() {
    assert_eq!(text(vm_call(builder_constructor(0).unwrap(), &[])), "");
    assert_eq!(text(vm_call(builder_constructor(1).unwrap(), &[i(16)])), "");
    assert!(text(vm_call(builder_constructor(1).unwrap(), &[i(-1)])).contains("NegativeArraySize"));
    assert!(text(vm_call(builder_constructor(1).unwrap(), &[Value::Null])).contains("NullPointer"));
    assert!(builder_constructor(2).is_none());
    assert_eq!(text(vm_call(string_constructor(0).unwrap(), &[])), "");
    assert!(string_constructor(1).is_none());
}
