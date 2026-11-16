#![cfg(test)]
use super::*;

fn ok(bytes: &[u8], encoding: Option<&str>) -> String {
    decode(bytes, encoding, "g.xml").unwrap()
}

fn err(bytes: &[u8], encoding: Option<&str>) -> (JavaException, String) {
    let e = decode(bytes, encoding, "g.xml").unwrap_err();
    (e.kind, e.message)
}

#[test]
fn an_argument_overrides_the_declaration() {
    let latin1 = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?><a>\xE4</a>";
    assert_eq!(
        ok(latin1, Some("iso-8859-1")),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><a>\u{E4}</a>"
    );
    assert_eq!(
        ok(latin1, Some("latin1")).chars().rev().nth(4),
        Some('\u{E4}')
    );
    // A UTF-8 file read as Latin-1: two chars per umlaut.
    assert_eq!(
        ok("<a>\u{E4}</a>".as_bytes(), Some("ISO-8859-1")),
        "<a>\u{C3}\u{A4}</a>"
    );
    // UTF-8 named: strict, the BOM skipped.
    assert_eq!(ok(b"\xEF\xBB\xBF<a/>", Some("utf-8")), "<a/>");
    let (kind, m) = err(latin1, Some("UTF-8"));
    assert_eq!(kind, JavaException::Io);
    assert_eq!(
            m,
            "org.xml.sax.SAXParseException; systemId: g.xml; lineNumber: 1; columnNumber: 42; Invalid byte 2 of 3-byte UTF-8 sequence."
        );
    // US-ASCII named: replaced, not refused.
    assert_eq!(ok(b"<a>\xE4</a>", Some("us-ascii")), "<a>\u{FFFD}</a>");
    assert_eq!(ok(b"<a>\xE4</a>", Some("KOI8-R")), "<a>\u{0414}</a>");
}

#[test]
fn an_argument_naming_no_charset() {
    assert_eq!(
        err(b"<a/>", Some("bogus")),
        (JavaException::UnsupportedEncoding, "BOGUS".into())
    );
    assert_eq!(
        err(b"<a/>", Some("x y")),
        (
            JavaException::Io,
            "org.xml.sax.SAXParseException; Invalid encoding name \"X Y\".".into()
        )
    );
    assert_eq!(
        err(b"<a/>", Some("1x")).1,
        "org.xml.sax.SAXParseException; Invalid encoding name \"1X\"."
    );
    assert_eq!(
        err(b"<a/>", Some("")).1,
        "org.xml.sax.SAXParseException; Invalid encoding name \"\"."
    );
    assert_eq!(
            err(b"<a/>", Some("iso-10646-ucs-2")).1,
            "org.xml.sax.SAXParseException; Given byte order for encoding \"ISO-10646-UCS-2\" is not supported."
        );
    // Every single-byte JDK charset decodes (an unmappable byte is
    // U+FFFD); a multi-byte one is refused.
    assert_eq!(
        ok(b"<a>\x80\x81</a>", Some("windows-1252")),
        "<a>\u{20AC}\u{FFFD}</a>"
    );
    assert_eq!(ok(&[0x4C, 0x81, 0x61, 0x6E], Some("ibm037")), "<a/>");
    assert_eq!(ok(b"<a>\x80</a>", Some("x-MacRoman")), "<a>\u{C4}</a>");
    let (kind, m) = err(b"<a/>", Some("Shift_JIS"));
    assert_eq!(kind, JavaException::UnsupportedOperation);
    assert!(m.contains("SHIFT_JIS"), "{m}");
}

#[test]
fn utf16_by_mark_or_pattern() {
    let le: Vec<u8> = "<a>\u{E4}</a>"
        .encode_utf16()
        .flat_map(u16::to_le_bytes)
        .collect();
    let be: Vec<u8> = "<a>\u{E4}</a>"
        .encode_utf16()
        .flat_map(u16::to_be_bytes)
        .collect();
    let marked = |m: &[u8], b: &[u8]| [m, b].concat();
    for enc in [None, Some("UTF-16"), Some("utf-16")] {
        assert_eq!(ok(&marked(b"\xFF\xFE", &le), enc), "<a>\u{E4}</a>");
        assert_eq!(ok(&marked(b"\xFE\xFF", &be), enc), "<a>\u{E4}</a>");
    }
    let pi_le: Vec<u8> = "<?xml version=\"1.0\"?><a/>"
        .encode_utf16()
        .flat_map(u16::to_le_bytes)
        .collect();
    let pi_be: Vec<u8> = "<?xml version=\"1.0\"?><a/>"
        .encode_utf16()
        .flat_map(u16::to_be_bytes)
        .collect();
    assert_eq!(ok(&pi_le, None), "<?xml version=\"1.0\"?><a/>");
    assert_eq!(ok(&pi_be, None), "<?xml version=\"1.0\"?><a/>");
    assert_eq!(ok(&pi_le, Some("UTF-16")), "<?xml version=\"1.0\"?><a/>");
    assert_eq!(ok(&be, Some("UTF-16")), "<a>\u{E4}</a>");
    // The JDK's UTF-16BE keeps the mark; an odd byte is replaced.
    assert_eq!(
        ok(&marked(b"\xFE\xFF", &be), Some("UTF-16BE")),
        "\u{FEFF}<a>\u{E4}</a>"
    );
    assert_eq!(ok(&le, Some("x-utf-16le")), "<a>\u{E4}</a>");
    assert_eq!(ok(b"\x00<\x00", Some("UTF-16BE")), "<\u{FFFD}");
}

#[test]
fn the_declaration_switches_a_utf8_stream() {
    let decl = |enc: &str, body: &[u8]| {
        [
            format!("<?xml version=\"1.0\" encoding=\"{enc}\"?>").as_bytes(),
            body,
        ]
        .concat()
    };
    assert!(ok(&decl("ISO-8859-1", b"<a>\xE4</a>"), None).ends_with("<a>\u{E4}</a>"));
    assert!(ok(&decl("latin1", b"<a>\xE4</a>"), None).ends_with("<a>\u{E4}</a>"));
    assert!(ok(&decl("utf8", b"<a>\xE4</a>"), None).ends_with("<a>\u{FFFD}</a>"));
    // After a UTF-8 mark too.
    assert!(ok(
        &[
            b"\xEF\xBB\xBF".as_slice(),
            &decl("ISO-8859-1", "<a>\u{E4}</a>".as_bytes())
        ]
        .concat(),
        None
    )
    .ends_with("<a>\u{C3}\u{A4}</a>"));
    assert!(ok(&decl("UTF-8", "<a>\u{E4}</a>".as_bytes()), None).ends_with("<a>\u{E4}</a>"));
    assert!(ok(&decl("us-ascii", b"<a>x</a>"), None).ends_with("<a>x</a>"));
    assert_eq!(
            err(&decl("US-ASCII", b"<a>\n\xE4</a>"), None).1,
            "org.xml.sax.SAXParseException; systemId: g.xml; lineNumber: 2; columnNumber: 1; Byte \"228\" is not a member of the (7-bit) ASCII character set."
        );
    assert_eq!(
        err(&decl("bogus", b"<a/>"), None),
        (JavaException::UnsupportedEncoding, "bogus".into())
    );
    assert_eq!(
            err(&decl("x y", b"<a/>"), None).1,
            "org.xml.sax.SAXParseException; systemId: g.xml; lineNumber: 1; columnNumber: 37; Invalid encoding name \"x y\"."
        );
    assert!(err(&decl("UTF-16", b"<a/>"), None)
        .1
        .ends_with("; Content is not allowed in prolog."));
    assert!(err(&decl("utf-32", b"<a/>"), None)
        .1
        .ends_with("; Content is not allowed in prolog."));
    assert!(err(&decl("ISO-10646-UCS-4", b"<a/>"), None)
        .1
        .ends_with("\"ISO-10646-UCS-4\" is not supported."));
    assert!(ok(&decl("Cp1252", b"<a>\x80</a>"), None).ends_with("<a>\u{20AC}</a>"));
    assert_eq!(
        err(&decl("EUC-JP", b"<a/>"), None).0,
        JavaException::UnsupportedOperation
    );
    // A declaration without an encoding, or not a declaration at all.
    assert_eq!(
        ok(b"<?xml version='1.0'?><a/>", None),
        "<?xml version='1.0'?><a/>"
    );
    assert_eq!(ok(b"<?xmlx?><a/>", None), "<?xmlx?><a/>");
    assert_eq!(ok(b"<?xml encoding?><a/>", None), "<?xml encoding?><a/>");
    assert_eq!(
        ok(b"<?xml encoding=x?><a/>", None),
        "<?xml encoding=x?><a/>"
    );
    assert_eq!(
        ok(b"<?xml encoding='latin1", None),
        "<?xml encoding='latin1"
    );
}

#[test]
fn xerces_utf8_errors() {
    let cases: [(&[u8], &str); 16] = [
        (b"\x80", "Invalid byte 1 of 1-byte"),
        (b"\xC0\x80", "Invalid byte 1 of 1-byte"),
        (b"\xC1\xBF", "Invalid byte 1 of 1-byte"),
        (b"\xC3", "Expected byte 2 of 2-byte"),
        (b"\xC3\x41", "Invalid byte 2 of 2-byte"),
        (b"\xE0\x80\x80", "Invalid byte 2 of 3-byte"),
        (b"\xED\xA0\x80", "Invalid byte 2 of 3-byte"),
        (b"\xE4\x41", "Invalid byte 2 of 3-byte"),
        (b"\xE4\xB8\x41", "Invalid byte 3 of 3-byte"),
        (b"\xE4\xB8", "Expected byte 3 of 3-byte"),
        (b"\xE4", "Expected byte 2 of 3-byte"),
        (b"\xF0\x80\x80\x80", "Invalid byte 2 of 4-byte"),
        (b"\xF0\x9F\x41", "Invalid byte 3 of 4-byte"),
        (b"\xF0\x9F\x98\x41", "Invalid byte 4 of 4-byte"),
        (b"\xF0\x9F\x98", "Expected byte 4 of 4-byte"),
        (b"\xF8", "Invalid byte 1 of 1-byte"),
    ];
    for (bytes, want) in cases {
        let input = [b"<a>\r\nx".as_slice(), bytes].concat();
        let m = err(&input, None).1;
        assert_eq!(
                m,
                format!("org.xml.sax.SAXParseException; systemId: g.xml; lineNumber: 2; columnNumber: 2; {want} UTF-8 sequence."),
                "{bytes:x?}"
            );
    }
    assert!(err(b"\xF0", None)
        .1
        .ends_with("Expected byte 2 of 4-byte UTF-8 sequence."));
    assert!(err(b"\xF0\x9F", None)
        .1
        .ends_with("Expected byte 3 of 4-byte UTF-8 sequence."));
    assert!(err(b"\xF4\x90\x80\x80", None)
        .1
        .ends_with("must not exceed 0x10 but found 0x11."));
    assert!(err(b"\xF5\x80\x80\x80", None)
        .1
        .ends_with("must not exceed 0x10 but found 0x14."));
    // Every well-formed sequence decodes, a supplementary one to a pair.
    let text = "a\u{E4}\u{4E2D}\u{1F600}\r\n\rz";
    assert_eq!(ok(text.as_bytes(), None), text);
    assert_eq!(ok(text.as_bytes(), Some("UTF-8")), text);
}

#[test]
fn charset_names() {
    assert_eq!(lookup("UTF8"), Lookup::Decoded(JdkCharset::Utf8));
    assert_eq!(
        lookup("ISO8859_15_FDIS"),
        Lookup::Decoded(JdkCharset::Table(3))
    );
    assert_eq!(lookup("ANSI_X3.4-1968"), Lookup::Decoded(JdkCharset::Ascii));
    assert_eq!(
        lookup("UnicodeBigUnmarked"),
        Lookup::Decoded(JdkCharset::Utf16Be)
    );
    assert!(matches!(
        lookup("windows-1252"),
        Lookup::Decoded(JdkCharset::Byte(_))
    ));
    assert_eq!(lookup("x-SJIS_0213"), Lookup::Undecoded);
    // Every single-byte alias is one the JDK has.
    for (n, _) in xml_charsets::NAMES {
        assert!(charsets::JDK_NAMES.binary_search(&n).is_ok(), "{n}");
    }
    assert_eq!(lookup("bogus"), Lookup::Unknown);
    // Every alias here is one the JDK has.
    for n in ASCII_NAMES.iter().chain(UTF16_NAMES.iter().map(|(n, _)| n)) {
        assert!(charsets::JDK_NAMES.binary_search(n).is_ok(), "{n}");
    }
    assert!(is_valid_iana("a.b_c-1"));
    assert!(!is_valid_iana("-a") && !is_valid_iana("") && !is_valid_iana("a b"));
    assert!(is_wide("UTF-16LE") && is_wide("UNICODE") && is_wide("UTF32") && !is_wide("UTF-8"));
}
