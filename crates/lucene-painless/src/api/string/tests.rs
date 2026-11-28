#![cfg(test)]
use super::*;
use crate::api::test_support::{call, elems, i, s, t, text};

#[test]
fn length_indexing_and_substrings_count_utf16_units() {
    let g = s("a\u{1F600}b");
    assert_eq!(t(&g, "length", &[]), "4");
    assert_eq!(t(&s(""), "isEmpty", &[]), "true");
    assert_eq!(t(&s("abc"), "charAt", &[i(1)]), "b");
    assert!(t(&s("abc"), "charAt", &[i(3)]).contains("StringIndexOutOfBounds"));
    assert_eq!(t(&s("hello"), "substring", &[i(1)]), "ello");
    assert_eq!(t(&s("hello"), "substring", &[i(1), i(3)]), "el");
    assert_eq!(t(&s("hello"), "subSequence", &[i(0), i(5)]), "hello");
    assert!(t(&s("hello"), "substring", &[i(3), i(2)]).contains("begin 3, end 2, length 5"));
    assert!(t(&s("hello"), "substring", &[i(-1), i(2)]).contains("StringIndexOutOfBounds"));
    assert_eq!(t(&g, "substring", &[i(0), i(1)]), "a");
}

#[test]
fn searches_return_java_indexes() {
    let h = s("banana");
    assert_eq!(t(&h, "indexOf", &[s("an")]), "1");
    assert_eq!(t(&h, "indexOf", &[s("an"), i(2)]), "3");
    assert_eq!(t(&h, "indexOf", &[s("zz")]), "-1");
    assert_eq!(t(&h, "lastIndexOf", &[s("an")]), "3");
    assert_eq!(t(&h, "lastIndexOf", &[s("an"), i(2)]), "1");
    assert_eq!(t(&h, "contains", &[s("nan")]), "true");
    assert_eq!(t(&h, "startsWith", &[s("ban")]), "true");
    assert_eq!(t(&h, "startsWith", &[s("nan"), i(2)]), "true");
    assert_eq!(t(&h, "startsWith", &[s("nan"), i(-1)]), "false");
    assert_eq!(t(&h, "startsWith", &[s("nana"), i(4)]), "false");
    assert_eq!(t(&h, "endsWith", &[s("ana")]), "true");
}

#[test]
fn equality_hashing_and_ordering() {
    assert_eq!(t(&s("a"), "equals", &[s("a")]), "true");
    assert_eq!(t(&s("a"), "equals", &[Value::Int(1)]), "false");
    assert_eq!(
        t(&s("Straße"), "equalsIgnoreCase", &[s("STRASSE")]),
        "false"
    );
    assert_eq!(t(&s("AbC"), "equalsIgnoreCase", &[s("aBc")]), "true");
    assert_eq!(t(&s("AbC"), "equalsIgnoreCase", &[s("aBcd")]), "false");
    assert_eq!(t(&s("AbC"), "equalsIgnoreCase", &[Value::Null]), "false");
    assert_eq!(t(&s("abc"), "hashCode", &[]), "96354");
    assert_eq!(t(&s("abc"), "toString", &[]), "abc");
    assert_eq!(t(&s("abc"), "compareTo", &[s("abd")]), "-1");
    assert_eq!(t(&s("abc"), "compareTo", &[s("ab")]), "1");
    assert_eq!(t(&s("ABC"), "compareToIgnoreCase", &[s("abd")]), "-1");
    assert_eq!(t(&s("abc"), "contentEquals", &[s("abc")]), "true");
    assert_eq!(java_compare("a", "b"), -1);
    assert_eq!(java_compare("abc", "abc"), 0);
}

#[test]
fn building_and_rewriting_text() {
    assert_eq!(t(&s("ab"), "concat", &[s("cd")]), "abcd");
    assert_eq!(
        t(
            &s("a-b-c"),
            "replace",
            &[Value::Char('-' as u16), Value::Char('+' as u16)]
        ),
        "a+b+c"
    );
    assert_eq!(t(&s("a-b-c"), "replace", &[s("-"), s("--")]), "a--b--c");
    assert_eq!(t(&s("ab"), "replace", &[s(""), s("_")]), "_a_b_");
    assert_eq!(t(&s("Hello"), "toLowerCase", &[]), "hello");
    assert_eq!(t(&s("Hello"), "toUpperCase", &[]), "HELLO");
    assert_eq!(t(&s("  x \t"), "trim", &[]), "x");
    assert_eq!(t(&s("  x "), "strip", &[]), "x");
    assert_eq!(elems(call(&s("ab"), "toCharArray", &[])), "[a, b]");
    assert_eq!(t(&s("a\u{1F600}"), "codePointAt", &[i(1)]), "128512");
    assert_eq!(t(&s("ab"), "codePointAt", &[i(1)]), "98");
    assert!(t(&s("ab"), "codePointAt", &[i(5)]).contains("Index 5 out of bounds for length 2"));
    assert_eq!(t(&s("a\u{1F600}b"), "codePointCount", &[i(0), i(4)]), "3");
}

#[test]
fn splitting_and_base64() {
    assert_eq!(
        elems(call(&s("a,b,,c"), "splitOnToken", &[s(",")])),
        "[a, b, , c]"
    );
    assert_eq!(
        elems(call(&s("a,b,c"), "splitOnToken", &[s(","), i(2)])),
        "[a, b,c]"
    );
    // unlike String.split, trailing empty strings stay, with no limit and with limit 0
    assert_eq!(
        elems(call(&s("a,b,,c,,"), "splitOnToken", &[s(",")])),
        "[a, b, , c, , ]"
    );
    assert_eq!(
        elems(call(&s("a,,"), "splitOnToken", &[s(","), i(0)])),
        "[a, , ]"
    );
    assert_eq!(elems(call(&s(""), "splitOnToken", &[s(",")])), "[]");
    assert_eq!(elems(call(&s(",,"), "splitOnToken", &[s(",")])), "[, , ]");
    assert_eq!(t(&s("hello"), "encodeBase64", &[]), "aGVsbG8=");
    assert_eq!(t(&s("aGVsbG8="), "decodeBase64", &[]), "hello");
    assert!(t(&s("***"), "decodeBase64", &[]).contains("IllegalArgument"));
    assert_eq!(base64_encode(b""), "");
    assert_eq!(base64_decode("aGk="), Ok(b"hi".to_vec()));
}

#[test]
fn region_matching() {
    let h = s("Hello World");
    assert_eq!(
        t(&h, "regionMatches", &[i(6), s("World"), i(0), i(5)]),
        "true"
    );
    assert_eq!(
        t(&h, "regionMatches", &[i(6), s("world"), i(0), i(5)]),
        "false"
    );
    assert_eq!(
        t(
            &h,
            "regionMatches",
            &[Value::Bool(true), i(6), s("world"), i(0), i(5)]
        ),
        "true"
    );
    assert_eq!(
        t(&h, "regionMatches", &[i(-1), s("H"), i(0), i(1)]),
        "false"
    );
    assert_eq!(
        t(&h, "regionMatches", &[i(6), s("World"), i(0), i(9)]),
        "false"
    );
}

#[test]
fn unsupported_and_unknown_methods() {
    assert!(t(&s("a"), "chars", &[]).contains("IntStream"));
    assert!(t(&s("a"), "matches", &[s("a")]).contains("matches"));
    assert!(t(&s("a"), "nope", &[]).contains("not found"));
    assert!(
        t(&Value::Null, "length", &[]).contains("NullPointer")
            || t(&Value::Null, "length", &[]).contains("null")
    );
}

#[test]
fn formatting() {
    assert_eq!(
        text(format("%d items", &[Value::Int(3)]).map(Value::string)),
        "3 items"
    );
    assert_eq!(
        text(format("%5.2f|", &[Value::Double(1.23456)]).map(Value::string)),
        " 1.23|"
    );
    assert_eq!(
        text(format("%s-%s", &[s("a"), Value::Null]).map(Value::string)),
        "a-null"
    );
    assert_eq!(
        text(format("%x", &[Value::Int(255)]).map(Value::string)),
        "ff"
    );
    assert_eq!(
        text(format("%05d", &[Value::Int(42)]).map(Value::string)),
        "00042"
    );
    assert_eq!(text(format("100%%", &[]).map(Value::string)), "100%");
    assert_eq!(format_fixed(2.5, 0), "3");
    assert_eq!(format_fixed(1.005, 1), "1.0");
    assert_eq!(format_fixed(-0.5, 0), "-1");
}

fn builder(init: &str) -> Value {
    let b = builder_constructor(0).unwrap();
    let v = crate::api::test_support::vm_call(b, &[]).unwrap();
    call(&v, "append", &[s(init)]).unwrap();
    v
}

#[test]
fn string_statics_and_constructors() {
    use crate::api::test_support::stat;
    let chars = Value::array(
        "char",
        vec![Value::Char('h' as u16), Value::Char('i' as u16)],
    );
    assert_eq!(
        text(stat(static_lookup, "valueOf", std::slice::from_ref(&chars))),
        "hi"
    );
    assert_eq!(text(stat(static_lookup, "copyValueOf", &[chars])), "hi");
    assert_eq!(text(stat(static_lookup, "valueOf", &[Value::Int(7)])), "7");
    assert!(text(stat(static_lookup, "copyValueOf", &[Value::Null])).contains("NullPointer"));
    let parts = Value::array("java.lang.String", vec![s("a"), s("b"), s("c")]);
    assert_eq!(text(stat(static_lookup, "join", &[s("-"), parts])), "a-b-c");
    assert_eq!(
        text(stat(
            static_lookup,
            "format",
            &[s("%s=%d"), Value::array("def", vec![s("n"), i(4)])]
        )),
        "n=4"
    );
    // a lone value that reached `String.format` through `def` is not the array the JVM wants
    assert!(text(stat(
        static_lookup,
        "format",
        &[s("%e"), Value::Double(0.5)]
    ))
    .contains("ClassCast"));
    assert_eq!(
        text(stat(static_lookup, "format", &[s("x"), Value::Null])),
        "x"
    );
    assert!(static_lookup("nope", 1).is_none());
    assert!(string_constructor(1).is_none());
    assert_eq!(
        text(crate::api::test_support::vm_call(
            string_constructor(0).unwrap(),
            &[]
        )),
        ""
    );
}

#[test]
fn string_builder_edits_in_place() {
    let b = builder("hello");
    assert_eq!(t(&b, "length", &[]), "5");
    assert_eq!(t(&b, "reverse", &[]), "olleh");
    assert_eq!(t(&b, "reverse", &[]), "hello");
    assert_eq!(t(&b, "insert", &[i(0), s(">> ")]), ">> hello");
    assert_eq!(t(&b, "delete", &[i(0), i(3)]), "hello");
    assert_eq!(t(&b, "deleteCharAt", &[i(0)]), "ello");
    assert_eq!(t(&b, "replace", &[i(0), i(1), s("J")]), "Jllo");
    assert_eq!(t(&b, "append", &[Value::Char('!' as u16)]), "Jllo!");
    assert_eq!(t(&b, "append", &[s("abcdef"), i(1), i(3)]), "Jllo!bc");
    assert_eq!(t(&b, "appendCodePoint", &[i(0x1F600)]), "Jllo!bc\u{1F600}");
    assert_eq!(t(&b, "setCharAt", &[i(0), Value::Char('H' as u16)]), "null");
    assert_eq!(t(&b, "toString", &[]), "Hllo!bc\u{1F600}");
    assert_eq!(t(&b, "setLength", &[i(2)]), "null");
    assert_eq!(t(&b, "toString", &[]), "Hl");
    assert_eq!(t(&b, "indexOf", &[s("l")]), "1");
    assert_eq!(t(&b, "lastIndexOf", &[s("l")]), "1");
    assert_eq!(t(&b, "charAt", &[i(0)]), "H");
    assert_eq!(t(&b, "substring", &[i(1)]), "l");
    assert_eq!(t(&b, "compareTo", &[builder("Hm")]), "-1");
}

#[test]
fn string_builder_range_errors() {
    let b = builder("abc");
    assert!(t(&b, "insert", &[i(9), s("x")]).contains("offset 9, length 3"));
    let r = t(&b, "delete", &[i(2), i(9)]);
    assert_eq!(r, "ab");
    assert!(t(&b, "delete", &[i(7), i(9)]).contains("StringIndexOutOfBounds"));
    assert!(t(&b, "deleteCharAt", &[i(7)]).contains("index 7,length 2"));
    assert!(t(&b, "setCharAt", &[i(7), Value::Char('x' as u16)]).contains("index 7,length 2"));
    assert!(t(&b, "setCharAt", &[i(0), i(1)]).contains("ClassCast"));
    assert!(builder_constructor(1).is_some());
}
