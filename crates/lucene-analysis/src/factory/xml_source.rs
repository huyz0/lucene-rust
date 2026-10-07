//! The hyphenation grammar as `HyphenationCompoundWordTokenFilterFactory.inform`
//! hands it to the JDK's SAX parser: an `org.xml.sax.InputSource` over the
//! resource's bytes, its encoding the factory's `encoding` argument ("if it's
//! null let xml parser decide"). This module is the decoding half of that
//! parser -- the JDK's Xerces picking a reader (`XMLEntityManager`) -- so the
//! text [`crate::compound::PatternParser`] reads is the text Java reads.
//!
//! With an `encoding` (externally specified, which makes Xerces ignore the
//! XML declaration's), the name is upper-cased; `UTF-8` is Xerces' strict
//! reader (a byte-order mark skipped), `UTF-16` its own reader (byte-order
//! mark, else `<?` in little-endian, else big-endian), a name that is not an
//! IANA name (a letter, then letters, digits, `.`, `_`, `-`) a parse error,
//! and any other name the JDK charset of that name through an
//! `InputStreamReader` (malformed and unmappable input replaced by U+FFFD) or,
//! when the JDK has none, Java's `UnsupportedEncodingException`.
//!
//! Without one, the first bytes decide: a UTF-8 byte-order mark, a UTF-16
//! one (either order), `<?` in UTF-16 without one, else UTF-8; in a UTF-8
//! stream the XML declaration's `encoding` then switches the reader --
//! `UTF-8` stays strict, `US-ASCII` (and its aliases) is Xerces' strict ASCII
//! reader, a 16- or 32-bit charset leaves bytes the parser cannot read, and
//! any other name is resolved as above, but as written (not upper-cased).
//!
//! Differs: a charset the JDK has but this port does not decode (all but
//! UTF-8, UTF-16, US-ASCII and the single-byte charsets of
//! [`crate::hunspell`]'s table, `charsets::NAMES`) is an
//! `UnsupportedOperationException` naming it; the XML declaration of a UTF-16
//! stream is not read (Xerces switches to the declared charset); and a
//! parse error reports its line and column at the offending character,
//! where Xerces reports the position its buffer had reached (often the
//! same).

use super::{FactoryError, JavaException};
use crate::hunspell::charsets;

/// `java.io.UnsupportedEncodingException`'s message for `name`.
fn unsupported_encoding(name: &str) -> FactoryError {
    FactoryError::new(JavaException::UnsupportedEncoding, name)
}

/// An `IOException` wrapping the SAX parser's `SAXParseException` (Lucene's
/// `PatternParser.parse` wraps every `SAXException`), as
/// `SAXParseException.toString()` prints it: located when the entity was
/// open.
fn sax_error(location: Option<(&str, usize, usize)>, message: &str) -> FactoryError {
    let text = match location {
        Some((system_id, line, column)) => format!(
            "org.xml.sax.SAXParseException; systemId: {system_id}; lineNumber: {line}; columnNumber: {column}; {message}"
        ),
        None => format!("org.xml.sax.SAXParseException; {message}"),
    };
    FactoryError::new(JavaException::Io, text)
}

/// A JDK charset this port decodes, through an `InputStreamReader`
/// (`CodingErrorAction.REPLACE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JdkCharset {
    Utf8,
    Latin1,
    Ascii,
    Utf16Be,
    Utf16Le,
    /// `charsets::TABLES[i]`.
    Table(usize),
}

/// The JDK's names and aliases of `US-ASCII`, lowercased.
const ASCII_NAMES: [&str; 14] = [
    "646",
    "ascii",
    "ascii7",
    "cp367",
    "csascii",
    "ibm367",
    "iso646-us",
    "iso_646.irv:1983",
    "iso_646.irv:1991",
    "us",
    "us-ascii",
    "ansi_x3.4-1968",
    "ansi_x3.4-1986",
    "iso-ir-6",
];

/// The JDK's names and aliases of `UTF-16BE` and `UTF-16LE`, lowercased.
const UTF16_NAMES: [(&str, JdkCharset); 8] = [
    ("utf-16be", JdkCharset::Utf16Be),
    ("utf_16be", JdkCharset::Utf16Be),
    ("x-utf-16be", JdkCharset::Utf16Be),
    ("unicodebigunmarked", JdkCharset::Utf16Be),
    ("utf-16le", JdkCharset::Utf16Le),
    ("utf_16le", JdkCharset::Utf16Le),
    ("x-utf-16le", JdkCharset::Utf16Le),
    ("unicodelittleunmarked", JdkCharset::Utf16Le),
];

/// What `Charset.forName` makes of a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lookup {
    Decoded(JdkCharset),
    /// The JDK has it; this port does not decode it.
    Undecoded,
    /// The JDK does not know it.
    Unknown,
}

fn lookup(name: &str) -> Lookup {
    let lower = name.to_ascii_lowercase();
    if let Ok(i) = charsets::NAMES.binary_search_by(|(k, _)| (*k).cmp(lower.as_str())) {
        return Lookup::Decoded(match charsets::NAMES[i].1 {
            0 => JdkCharset::Utf8,
            1 => JdkCharset::Latin1,
            t => JdkCharset::Table(t - 2),
        });
    }
    if ASCII_NAMES.contains(&lower.as_str()) {
        return Lookup::Decoded(JdkCharset::Ascii);
    }
    if let Some((_, cs)) = UTF16_NAMES.iter().find(|(n, _)| *n == lower) {
        return Lookup::Decoded(*cs);
    }
    if charsets::JDK_NAMES.binary_search(&lower.as_str()).is_ok() {
        Lookup::Undecoded
    } else {
        Lookup::Unknown
    }
}

/// `XMLChar.isValidIANAEncoding`: a letter, then letters, digits, `.`,
/// `_` and `-`.
fn is_valid_iana(name: &str) -> bool {
    let mut bytes = name.bytes();
    matches!(bytes.next(), Some(b) if b.is_ascii_alphabetic())
        && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Whether `upper` names a charset of 16- or 32-bit units, which a stream
/// read as bytes cannot switch to.
fn is_wide(upper: &str) -> bool {
    upper.starts_with("UTF-16")
        || upper.starts_with("UTF_16")
        || upper.starts_with("UTF-32")
        || upper.starts_with("UTF_32")
        || upper.starts_with("UNICODE")
        || upper == "UTF16"
        || upper == "UTF32"
}

/// Java chars, with the position of the next one for a parse error.
struct Chars {
    units: Vec<u16>,
    line: usize,
    column: usize,
    after_cr: bool,
}

impl Chars {
    fn new() -> Self {
        Chars {
            units: Vec::new(),
            line: 1,
            column: 1,
            after_cr: false,
        }
    }

    fn push(&mut self, c: u16) {
        self.units.push(c);
        match c {
            0x0A if self.after_cr => {}
            0x0A | 0x0D => {
                self.line += 1;
                self.column = 1;
            }
            _ => self.column += 1,
        }
        self.after_cr = c == 0x0D;
    }

    fn push_char(&mut self, c: char) {
        let mut buf = [0u16; 2];
        for &u in c.encode_utf16(&mut buf).iter() {
            self.push(u);
        }
    }
}

/// Xerces' `UTF8Reader`: every malformed sequence is an error, with
/// Xerces' messages.
fn utf8_strict(bytes: &[u8], system_id: &str) -> Result<Vec<u16>, FactoryError> {
    let mut out = Chars::new();
    let mut i = 0;
    let fail = |out: &Chars, message: String| {
        Err(sax_error(Some((system_id, out.line, out.column)), &message))
    };
    let invalid = |n: usize, of: usize| format!("Invalid byte {n} of {of}-byte UTF-8 sequence.");
    let expected = |n: usize, of: usize| format!("Expected byte {n} of {of}-byte UTF-8 sequence.");
    let cont = |b: u8| b & 0xC0 == 0x80;
    while i < bytes.len() {
        let b0 = bytes[i];
        let at = |k: usize| bytes.get(i + k).copied();
        if b0 < 0x80 {
            out.push(u16::from(b0));
            i += 1;
        } else if b0 & 0xE0 == 0xC0 && b0 & 0x1E != 0 {
            let Some(b1) = at(1) else {
                return fail(&out, expected(2, 2));
            };
            if !cont(b1) {
                return fail(&out, invalid(2, 2));
            }
            out.push((u16::from(b0 & 0x1F) << 6) | u16::from(b1 & 0x3F));
            i += 2;
        } else if b0 & 0xF0 == 0xE0 {
            let Some(b1) = at(1) else {
                return fail(&out, expected(2, 3));
            };
            if !cont(b1) || (b0 == 0xED && b1 >= 0xA0) || (b0 & 0x0F == 0 && b1 & 0x20 == 0) {
                return fail(&out, invalid(2, 3));
            }
            let Some(b2) = at(2) else {
                return fail(&out, expected(3, 3));
            };
            if !cont(b2) {
                return fail(&out, invalid(3, 3));
            }
            out.push(
                (u16::from(b0 & 0x0F) << 12) | (u16::from(b1 & 0x3F) << 6) | u16::from(b2 & 0x3F),
            );
            i += 3;
        } else if b0 & 0xF8 == 0xF0 {
            let Some(b1) = at(1) else {
                return fail(&out, expected(2, 4));
            };
            if !cont(b1) || (b1 & 0x30 == 0 && b0 & 0x07 == 0) {
                return fail(&out, invalid(2, 4));
            }
            let Some(b2) = at(2) else {
                return fail(&out, expected(3, 4));
            };
            if !cont(b2) {
                return fail(&out, invalid(3, 4));
            }
            let Some(b3) = at(3) else {
                return fail(&out, expected(4, 4));
            };
            if !cont(b3) {
                return fail(&out, invalid(4, 4));
            }
            let planes = ((b0 << 2) & 0x1C) | ((b1 >> 4) & 0x03);
            if planes > 0x10 {
                return fail(
                    &out,
                    format!(
                        "High surrogate bits in UTF-8 sequence must not exceed 0x10 but found 0x{planes:x}."
                    ),
                );
            }
            let c = (u32::from(b0 & 0x07) << 18)
                | (u32::from(b1 & 0x3F) << 12)
                | (u32::from(b2 & 0x3F) << 6)
                | u32::from(b3 & 0x3F);
            out.push_char(char::from_u32(c).unwrap_or('\u{FFFD}'));
            i += 4;
        } else {
            return fail(&out, invalid(1, 1));
        }
    }
    Ok(out.units)
}

/// Xerces' `ASCIIReader`: a byte above `0x7F` is an error.
fn ascii_strict(bytes: &[u8], system_id: &str) -> Result<Vec<u16>, FactoryError> {
    let mut out = Chars::new();
    for &b in bytes {
        if b >= 0x80 {
            return Err(sax_error(
                Some((system_id, out.line, out.column)),
                &format!("Byte \"{b}\" is not a member of the (7-bit) ASCII character set."),
            ));
        }
        out.push(u16::from(b));
    }
    Ok(out.units)
}

fn utf16(bytes: &[u8], big_endian: bool) -> Vec<u16> {
    bytes
        .chunks(2)
        .map(|p| match *p {
            [a, b] if big_endian => u16::from_be_bytes([a, b]),
            [a, b] => u16::from_le_bytes([a, b]),
            _ => 0xFFFD,
        })
        .collect()
}

/// `new InputStreamReader(stream, charset)`: malformed and unmappable input
/// replaced.
fn replacing(bytes: &[u8], charset: JdkCharset) -> Vec<u16> {
    match charset {
        JdkCharset::Utf8 => String::from_utf8_lossy(bytes).encode_utf16().collect(),
        JdkCharset::Latin1 => bytes.iter().map(|&b| u16::from(b)).collect(),
        JdkCharset::Ascii => bytes
            .iter()
            .map(|&b| if b < 0x80 { u16::from(b) } else { 0xFFFD })
            .collect(),
        JdkCharset::Utf16Be => utf16(bytes, true),
        JdkCharset::Utf16Le => utf16(bytes, false),
        JdkCharset::Table(t) => {
            let table = charsets::TABLES[t];
            bytes
                .iter()
                .map(|&b| {
                    if b < 0x80 {
                        u16::from(b)
                    } else {
                        table[usize::from(b - 0x80)]
                    }
                })
                .collect()
        }
    }
}

/// Xerces' reader for `UTF-16`: the byte-order mark (skipped), else `<?`
/// little-endian, else big-endian.
fn xerces_utf16(bytes: &[u8]) -> Vec<u16> {
    match bytes {
        [0xFE, 0xFF, rest @ ..] => utf16(rest, true),
        [0xFF, 0xFE, rest @ ..] => utf16(rest, false),
        [0x3C, 0x00, 0x3F, 0x00, ..] => utf16(bytes, false),
        _ => utf16(bytes, true),
    }
}

/// A charset named to Xerces outside its own readers: the JDK's, or
/// `UnsupportedEncodingException` naming it as given.
fn jdk_reader(bytes: &[u8], name: &str) -> Result<Vec<u16>, FactoryError> {
    match lookup(name) {
        Lookup::Decoded(cs) => Ok(replacing(bytes, cs)),
        Lookup::Undecoded => Err(FactoryError::new(
            JavaException::UnsupportedOperation,
            format!("charset {name} is not decoded by this port"),
        )),
        Lookup::Unknown => Err(unsupported_encoding(name)),
    }
}

/// The `encoding` pseudo-attribute of an XML declaration at the start of
/// `bytes`, read as ASCII.
fn declared_encoding(bytes: &[u8]) -> Option<String> {
    let rest = bytes.strip_prefix(b"<?xml")?;
    if !rest.first().is_some_and(|b| b.is_ascii_whitespace()) {
        return None;
    }
    let end = rest.windows(2).position(|w| w == b"?>")?;
    let decl = std::str::from_utf8(&rest[..end]).ok()?;
    let at = decl.find("encoding")?;
    let after = decl[at + "encoding".len()..].trim_start();
    let after = after.strip_prefix('=')?.trim_start();
    let quote = after.chars().next().filter(|&q| q == '"' || q == '\'')?;
    let value = &after[1..];
    Some(value[..value.find(quote)?].to_string())
}

/// The text of `bytes` as the JDK's SAX parser reads it, `encoding` the
/// factory's argument and `system_id` the resource name (Java:
/// `is.setSystemId(hypFile)`).
pub(crate) fn decode(
    bytes: &[u8],
    encoding: Option<&str>,
    system_id: &str,
) -> Result<String, FactoryError> {
    let units = match encoding {
        Some(name) => externally_specified(bytes, name, system_id)?,
        None => detected(bytes, system_id)?,
    };
    Ok(String::from_utf16_lossy(&units))
}

fn externally_specified(
    bytes: &[u8],
    name: &str,
    system_id: &str,
) -> Result<Vec<u16>, FactoryError> {
    let upper = name.to_uppercase();
    match upper.as_str() {
        "UTF-8" => utf8_strict(
            bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes),
            system_id,
        ),
        "UTF-16" => Ok(xerces_utf16(bytes)),
        "ISO-10646-UCS-2" | "ISO-10646-UCS-4" => Err(sax_error(
            None,
            &format!("Given byte order for encoding \"{upper}\" is not supported."),
        )),
        _ if !is_valid_iana(&upper) => Err(sax_error(
            None,
            &format!("Invalid encoding name \"{upper}\"."),
        )),
        _ => jdk_reader(bytes, &upper),
    }
}

fn detected(bytes: &[u8], system_id: &str) -> Result<Vec<u16>, FactoryError> {
    let body = match bytes {
        [0xFE, 0xFF, ..]
        | [0xFF, 0xFE, ..]
        | [0x3C, 0x00, 0x3F, 0x00, ..]
        | [0x00, 0x3C, 0x00, 0x3F, ..] => {
            return Ok(xerces_utf16(bytes));
        }
        [0xEF, 0xBB, 0xBF, rest @ ..] => rest,
        _ => bytes,
    };
    let Some(name) = declared_encoding(body) else {
        return utf8_strict(body, system_id);
    };
    let upper = name.to_uppercase();
    // The declaration is read before the switch: its position is past it.
    let decl_end = body
        .windows(2)
        .position(|w| w == b"?>")
        .map_or(1, |p| p + 3);
    let located = |message: String| Err(sax_error(Some((system_id, 1, decl_end)), &message));
    if upper == "UTF-8" {
        utf8_strict(body, system_id)
    } else if !is_valid_iana(&name) {
        located(format!("Invalid encoding name \"{name}\"."))
    } else if upper == "ISO-10646-UCS-2" || upper == "ISO-10646-UCS-4" {
        located(format!(
            "Given byte order for encoding \"{upper}\" is not supported."
        ))
    } else if is_wide(&upper) {
        located("Content is not allowed in prolog.".to_string())
    } else if ASCII_NAMES.contains(&name.to_ascii_lowercase().as_str()) {
        ascii_strict(body, system_id)
    } else {
        jdk_reader(body, &name)
    }
}

#[cfg(test)]
mod tests {
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
        let (kind, m) = err(b"<a/>", Some("windows-1252"));
        assert_eq!(kind, JavaException::UnsupportedOperation);
        assert!(m.contains("WINDOWS-1252"), "{m}");
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
        assert_eq!(
            err(&decl("Cp1252", b"<a/>"), None).0,
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
        assert_eq!(lookup("windows-1252"), Lookup::Undecoded);
        assert_eq!(lookup("bogus"), Lookup::Unknown);
        // Every alias here is one the JDK has.
        for n in ASCII_NAMES.iter().chain(UTF16_NAMES.iter().map(|(n, _)| n)) {
            assert!(charsets::JDK_NAMES.binary_search(n).is_ok(), "{n}");
        }
        assert!(is_valid_iana("a.b_c-1"));
        assert!(!is_valid_iana("-a") && !is_valid_iana("") && !is_valid_iana("a b"));
        assert!(is_wide("UTF-16LE") && is_wide("UNICODE") && is_wide("UTF32") && !is_wide("UTF-8"));
    }
}
