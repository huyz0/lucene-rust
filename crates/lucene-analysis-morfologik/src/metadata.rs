//! Morfologik's `DictionaryMetadata` and `DictionaryAttribute`: the `.info`
//! file beside a dictionary -- its separator, charset, sequence encoder and
//! conversions, each attribute validated as Morfologik validates it.
//!
//! Charsets: `UTF-8` (every dictionary Lucene loads), `ISO-8859-1` and
//! `US-ASCII`, under their canonical names and JDK 21's `aliases()`
//! (checked against `Charset.forName` in `charsets.tsv`); any other charset
//! is refused when the metadata is read (Java supports the JDK's whole
//! set).

use crate::properties;
use crate::MorfologikError;

/// `EncoderType`: how a dictionary entry spells its base form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    /// `SUFFIX`: `TrimSuffixEncoder`.
    Suffix,
    /// `PREFIX`: `TrimPrefixAndSuffixEncoder`.
    Prefix,
    /// `INFIX`: `TrimInfixAndSuffixEncoder`.
    Infix,
    /// `NONE`: `NoEncoder`.
    None,
}

impl EncoderType {
    /// `ISequenceEncoder.prefixBytes()`.
    pub fn prefix_bytes(self) -> usize {
        match self {
            EncoderType::Suffix => 1,
            EncoderType::Prefix => 2,
            EncoderType::Infix => 3,
            EncoderType::None => 0,
        }
    }
}

/// A dictionary charset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Charset {
    /// `UTF-8`.
    Utf8,
    /// `ISO-8859-1`.
    Latin1,
    /// `US-ASCII`.
    Ascii,
}

impl Charset {
    /// `Charset.forName` for the supported charsets: the canonical name or
    /// one of JDK 21's `aliases()` (the same in JDK 25), ignoring ASCII case.
    pub fn for_name(name: &str) -> Option<Charset> {
        const UTF8: [&str; 3] = ["UTF-8", "unicode-1-1-utf-8", "UTF8"];
        const ASCII: [&str; 14] = [
            "US-ASCII",
            "cp367",
            "ANSI_X3.4-1986",
            "us",
            "646",
            "ISO646-US",
            "ISO_646.irv:1991",
            "csASCII",
            "IBM367",
            "ascii7",
            "ANSI_X3.4-1968",
            "iso_646.irv:1983",
            "ASCII",
            "iso-ir-6",
        ];
        const LATIN1: [&str; 15] = [
            "ISO-8859-1",
            "latin1",
            "ISO8859-1",
            "iso-ir-100",
            "ISO_8859-1:1987",
            "ISO8859_1",
            "819",
            "l1",
            "ISO_8859-1",
            "8859_1",
            "IBM-819",
            "cp819",
            "ISO_8859_1",
            "csISOLatin1",
            "IBM819",
        ];
        let is = |names: &[&str]| names.iter().any(|n| n.eq_ignore_ascii_case(name));
        if is(&UTF8) {
            Some(Charset::Utf8)
        } else if is(&ASCII) {
            Some(Charset::Ascii)
        } else if is(&LATIN1) {
            Some(Charset::Latin1)
        } else {
            None
        }
    }

    /// `CharsetEncoder.encode` with `REPORT`: `None` for a character the
    /// charset cannot map (or an unpaired surrogate).
    pub fn encode(self, units: &[u16]) -> Option<Vec<u8>> {
        match self {
            Charset::Utf8 => String::from_utf16(units).ok().map(String::into_bytes),
            Charset::Latin1 => units.iter().map(|&u| u8::try_from(u).ok()).collect(),
            Charset::Ascii => units
                .iter()
                .map(|&u| u8::try_from(u).ok().filter(u8::is_ascii))
                .collect(),
        }
    }

    /// [`Charset::encode`] of a `&str` into `out` (cleared first): `false`
    /// for a character the charset cannot map.
    pub fn encode_str(self, text: &str, out: &mut Vec<u8>) -> bool {
        out.clear();
        match self {
            Charset::Utf8 => {
                out.extend_from_slice(text.as_bytes());
                true
            }
            Charset::Latin1 | Charset::Ascii => {
                let limit = if self == Charset::Latin1 { 0xFF } else { 0x7F };
                for c in text.chars() {
                    match u8::try_from(u32::from(c)) {
                        Ok(b) if u32::from(b) <= limit => out.push(b),
                        _ => return false,
                    }
                }
                true
            }
        }
    }

    /// [`Charset::decode`] to text, into `out` (cleared first): `false` for
    /// malformed input.
    pub fn decode_into(self, bytes: &[u8], out: &mut String) -> bool {
        out.clear();
        match self {
            Charset::Utf8 | Charset::Ascii => match std::str::from_utf8(bytes) {
                Ok(s) if self == Charset::Utf8 || s.is_ascii() => {
                    out.push_str(s);
                    true
                }
                _ => false,
            },
            Charset::Latin1 => {
                out.extend(bytes.iter().map(|&b| char::from(b)));
                true
            }
        }
    }

    /// `CharsetDecoder.decode` with `REPORT`: `None` for malformed input.
    pub fn decode(self, bytes: &[u8]) -> Option<Vec<u16>> {
        match self {
            Charset::Utf8 => std::str::from_utf8(bytes)
                .ok()
                .map(|s| s.encode_utf16().collect()),
            Charset::Latin1 => Some(bytes.iter().map(|&b| u16::from(b)).collect()),
            Charset::Ascii => bytes
                .iter()
                .map(|&b| b.is_ascii().then_some(u16::from(b)))
                .collect(),
        }
    }
}

/// `DictionaryAttribute`'s property names, in the enum's order (the order
/// Morfologik validates them in).
const ATTRIBUTES: [&str; 20] = [
    "fsa.dict.separator",
    "fsa.dict.encoding",
    "fsa.dict.frequency-included",
    "fsa.dict.speller.ignore-numbers",
    "fsa.dict.speller.ignore-punctuation",
    "fsa.dict.speller.ignore-camel-case",
    "fsa.dict.speller.ignore-all-uppercase",
    "fsa.dict.speller.ignore-diacritics",
    "fsa.dict.speller.convert-case",
    "fsa.dict.speller.runon-words",
    "fsa.dict.speller.locale",
    "fsa.dict.encoder",
    "fsa.dict.input-conversion",
    "fsa.dict.output-conversion",
    "fsa.dict.speller.replacement-pairs",
    "fsa.dict.speller.equivalent-chars",
    "fsa.dict.license",
    "fsa.dict.author",
    "fsa.dict.created",
    // Not an attribute: marks the end of the list for the index arithmetic.
    "",
];

fn iae(message: impl Into<String>) -> MorfologikError {
    MorfologikError::new(format!("IllegalArgumentException: {}", message.into()))
}

/// `DictionaryAttribute.booleanValue`.
fn boolean_value(value: &str) -> Result<bool, MorfologikError> {
    match value.to_lowercase().as_str() {
        "true" | "yes" | "on" => Ok(true),
        "false" | "no" | "off" => Ok(false),
        v => Err(iae(format!("Not a boolean value: {v}"))),
    }
}

/// `value.split(",\\s*")` then each `pair.trim().split(" ")`: Morfologik's
/// pair lists; `None` for an entry that is not exactly two parts.
fn pairs(value: &str) -> Option<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut parts: Vec<&str> = Vec::new();
    let mut rest = value;
    while let Some(i) = rest.find(',') {
        parts.push(&rest[..i]);
        rest = rest[i..][1..].trim_start_matches(|c: char| {
            matches!(c, ' ' | '\t' | '\n' | '\u{b}' | '\u{c}' | '\r')
        });
    }
    parts.push(rest);
    // Java's split drops trailing empty strings.
    while parts.len() > 1 && parts.last().is_some_and(|p| p.is_empty()) {
        parts.pop();
    }
    for p in parts {
        let p = p.trim_matches(|c: char| c <= ' ');
        let mut two: Vec<&str> = p.split(' ').collect();
        while two.len() > 1 && two.last().is_some_and(|s| s.is_empty()) {
            two.pop();
        }
        if two.len() != 2 {
            return None;
        }
        out.push((two[0].to_string(), two[1].to_string()));
    }
    Some(out)
}

/// Conversion pairs `(from, to)` over UTF-16 units, in order.
pub type Conversions = Vec<(Vec<u16>, Vec<u16>)>;

/// A conversion attribute: pairs in order, each input once.
fn conversion(name: &str, value: &str) -> Result<Conversions, MorfologikError> {
    let pairs = pairs(value).ok_or_else(|| {
        iae(format!(
            "Attribute {name} is not in the proper format: {value}"
        ))
    })?;
    let mut out: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();
    for (a, b) in pairs {
        let a: Vec<u16> = a.encode_utf16().collect();
        if out.iter().any(|(k, _)| *k == a) {
            return Err(iae(format!(
                "Input conversion cannot specify different values for the same input string: {}",
                String::from_utf16_lossy(&a)
            )));
        }
        out.push((a, b.encode_utf16().collect()));
    }
    Ok(out)
}

/// `morfologik.stemming.DictionaryMetadata`, the parts lookups use.
#[derive(Debug, Clone)]
pub struct DictionaryMetadata {
    /// `getSeparatorAsChar()`.
    pub separator_char: u16,
    /// `getSeparator()`: the separator in the dictionary's charset.
    pub separator: u8,
    /// The dictionary's charset.
    pub charset: Charset,
    /// `getSequenceEncoderType()`.
    pub encoder: EncoderType,
    /// `getInputConversionPairs()`, in order.
    pub input_conversion: Conversions,
    /// `getOutputConversionPairs()`, in order.
    pub output_conversion: Conversions,
}

impl DictionaryMetadata {
    /// `DictionaryMetadata.read(InputStream)`: the `.info` text (UTF-8).
    pub fn read(text: &str) -> Result<DictionaryMetadata, MorfologikError> {
        let props = properties::load(text)?;
        let get = |k: &str| {
            props
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.as_str())
        };
        if get("fsa.dict.encoder").is_none() {
            let deprecated = [
                "fsa.dict.uses-suffixes",
                "fsa.dict.uses-infixes",
                "fsa.dict.uses-prefixes",
            ]
            .iter()
            .any(|k| get(k).is_some());
            let flag = |k: &str, d: &str| get(k).unwrap_or(d).eq_ignore_ascii_case("true");
            let encoder = if flag("fsa.dict.uses-infixes", "false") {
                "INFIX"
            } else if flag("fsa.dict.uses-prefixes", "false") {
                "PREFIX"
            } else if flag("fsa.dict.uses-suffixes", "true") {
                "SUFFIX"
            } else {
                "NONE"
            };
            return Err(MorfologikError::new(if deprecated {
                format!("IOException: Deprecated encoder keys in metadata. Use fsa.dict.encoder={encoder}")
            } else {
                format!("IOException: Use an explicit fsa.dict.encoder={encoder} metadata key: ")
            }));
        }
        for (k, _) in &props {
            if !ATTRIBUTES[..19].contains(&k.as_str()) {
                return Err(iae(format!("No attribute for property: {k}")));
            }
        }
        let mut separator_char = None;
        let mut charset = None;
        let mut encoder = None;
        let mut input_conversion = Vec::new();
        let mut output_conversion = Vec::new();
        // Java: the defaults, then the file's values, in the enum's order.
        for &name in &ATTRIBUTES[..19] {
            let default = match name {
                "fsa.dict.frequency-included" => Some("false"),
                n if n.starts_with("fsa.dict.speller.ignore-")
                    || n == "fsa.dict.speller.convert-case"
                    || n == "fsa.dict.speller.runon-words" =>
                {
                    Some("true")
                }
                _ => None,
            };
            let Some(value) = get(name).or(default) else {
                continue;
            };
            match name {
                "fsa.dict.separator" => {
                    let units: Vec<u16> = value.encode_utf16().collect();
                    if units.len() != 1 {
                        return Err(iae(
                            "Attribute fsa.dict.separator must be a single character.",
                        ));
                    }
                    if (0xD800..=0xDFFF).contains(&units[0]) {
                        return Err(iae(format!(
                            "Field separator character cannot be part of a surrogate pair: {value}"
                        )));
                    }
                    separator_char = Some(units[0]);
                }
                "fsa.dict.encoding" => {
                    charset = Some(Charset::for_name(value).ok_or_else(|| {
                        MorfologikError::new(format!("UnsupportedCharsetException: {value}"))
                    })?);
                }
                "fsa.dict.encoder" => {
                    let v = value.trim_matches(|c: char| c <= ' ');
                    encoder = Some(match v.to_uppercase().as_str() {
                        "SUFFIX" => EncoderType::Suffix,
                        "PREFIX" => EncoderType::Prefix,
                        "INFIX" => EncoderType::Infix,
                        "NONE" => EncoderType::None,
                        _ => {
                            return Err(iae(format!(
                                "Invalid encoder name '{v}', only these coders are valid: [SUFFIX, PREFIX, INFIX, NONE]"
                            )))
                        }
                    });
                }
                "fsa.dict.input-conversion" => input_conversion = conversion(name, value)?,
                "fsa.dict.output-conversion" => output_conversion = conversion(name, value)?,
                "fsa.dict.speller.replacement-pairs" => {
                    pairs(value).ok_or_else(|| {
                        iae(format!(
                            "Attribute {name} is not in the proper format: {value}"
                        ))
                    })?;
                }
                "fsa.dict.speller.equivalent-chars" => {
                    let ok = pairs(value).is_some_and(|ps| {
                        ps.iter().all(|(a, b)| {
                            a.encode_utf16().count() == 1 && b.encode_utf16().count() == 1
                        })
                    });
                    if !ok {
                        return Err(iae(format!(
                            "Attribute {name} is not in the proper format: {value}"
                        )));
                    }
                }
                "fsa.dict.speller.locale"
                | "fsa.dict.license"
                | "fsa.dict.author"
                | "fsa.dict.created" => {}
                _ => {
                    boolean_value(value)?;
                }
            }
        }
        let missing: Vec<&str> = [
            ("SEPARATOR", separator_char.is_none()),
            ("ENCODING", charset.is_none()),
            ("ENCODER", encoder.is_none()),
        ]
        .iter()
        .filter(|(_, m)| *m)
        .map(|(n, _)| *n)
        .collect();
        let (Some(separator_char), Some(charset), Some(encoder)) =
            (separator_char, charset, encoder)
        else {
            return Err(iae(format!(
                "At least one the required attributes was not provided: [{}]",
                missing.join(", ")
            )));
        };
        let encoded = charset.encode(&[separator_char]).ok_or_else(|| {
            iae(format!(
                "Separator character cannot be converted to a byte in {}: {}",
                get("fsa.dict.encoding").unwrap_or(""),
                String::from_utf16_lossy(&[separator_char])
            ))
        })?;
        let [separator] = encoded[..] else {
            return Err(iae(format!(
                "Separator character is not a single byte in encoding {}: {}",
                get("fsa.dict.encoding").unwrap_or(""),
                String::from_utf16_lossy(&[separator_char])
            )));
        };
        Ok(DictionaryMetadata {
            separator_char,
            separator,
            charset,
            encoder,
            input_conversion,
            output_conversion,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OK: &str = "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX\n";

    #[test]
    fn reads_metadata() {
        let m = DictionaryMetadata::read(OK).unwrap();
        assert_eq!(m.separator, b'+');
        assert_eq!(m.separator_char, u16::from(b'+'));
        assert_eq!(m.charset, Charset::Utf8);
        assert_eq!(m.encoder, EncoderType::Suffix);
        let m = DictionaryMetadata::read(&format!(
            "{OK}fsa.dict.input-conversion=a b, cc d\nfsa.dict.output-conversion=x y\nfsa.dict.speller.locale=pl\nfsa.dict.speller.replacement-pairs=a b, a c\nfsa.dict.speller.equivalent-chars=a b\nfsa.dict.speller.ignore-numbers=yes\nfsa.dict.author=me"
        ))
        .unwrap();
        assert_eq!(m.input_conversion.len(), 2);
        assert_eq!(m.output_conversion.len(), 1);
        for (enc, t) in [
            ("PREFIX", EncoderType::Prefix),
            (" infix ", EncoderType::Infix),
            ("none", EncoderType::None),
        ] {
            let m = DictionaryMetadata::read(&format!(
                "fsa.dict.separator=+\nfsa.dict.encoding=iso-8859-1\nfsa.dict.encoder={enc}"
            ))
            .unwrap();
            assert_eq!(m.encoder, t);
            assert_eq!(m.charset, Charset::Latin1);
        }
        assert_eq!(EncoderType::Infix.prefix_bytes(), 3);
        assert_eq!(EncoderType::None.prefix_bytes(), 0);
        assert_eq!(EncoderType::Prefix.prefix_bytes(), 2);
        assert_eq!(EncoderType::Suffix.prefix_bytes(), 1);
    }

    #[test]
    fn refuses_bad_metadata() {
        let bad = [
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.uses-prefixes=true",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.uses-infixes=true\nfsa.dict.uses-suffixes=false",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.uses-suffixes=false",
            "fsa.dict.separator=++\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX",
            "fsa.dict.separator=\\uD800\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX",
            "fsa.dict.separator=+\nfsa.dict.encoding=koi8-r\nfsa.dict.encoder=SUFFIX",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=BOGUS",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX\nfsa.dict.unknown=1",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX\nfsa.dict.frequency-included=maybe",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX\nfsa.dict.input-conversion=a",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX\nfsa.dict.input-conversion=a b, a c",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX\nfsa.dict.speller.replacement-pairs=a",
            "fsa.dict.separator=+\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX\nfsa.dict.speller.equivalent-chars=ab c",
            "fsa.dict.separator=ż\nfsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX",
            "fsa.dict.separator=ż\nfsa.dict.encoding=us-ascii\nfsa.dict.encoder=SUFFIX",
            "fsa.dict.encoding=utf-8\nfsa.dict.encoder=SUFFIX",
            "x=\\u1",
        ];
        for b in bad {
            assert!(DictionaryMetadata::read(b).is_err(), "{b}");
        }
        let e = DictionaryMetadata::read("fsa.dict.encoder=SUFFIX").unwrap_err();
        assert!(
            e.message().ends_with("[SEPARATOR, ENCODING]"),
            "{}",
            e.message()
        );
    }

    #[test]
    fn charsets() {
        assert_eq!(Charset::Ascii.encode(&[0x41, 0x80]), None);
        assert_eq!(Charset::Ascii.encode(&[0x41]), Some(vec![0x41]));
        assert_eq!(Charset::Latin1.encode(&[0xE9]), Some(vec![0xE9]));
        assert_eq!(Charset::Latin1.encode(&[0x141]), None);
        assert_eq!(Charset::Utf8.encode(&[0xD800]), None);
        assert_eq!(Charset::Utf8.decode(&[0xC3, 0xA9]), Some(vec![0xE9]));
        assert_eq!(Charset::Utf8.decode(&[0xC3]), None);
        assert_eq!(Charset::Latin1.decode(&[0xE9]), Some(vec![0xE9]));
        assert_eq!(Charset::Ascii.decode(&[0xE9]), None);
        let mut b = vec![1];
        assert!(Charset::Utf8.encode_str("é", &mut b) && b == [0xC3, 0xA9]);
        assert!(Charset::Latin1.encode_str("é", &mut b) && b == [0xE9]);
        assert!(!Charset::Latin1.encode_str("ł", &mut b));
        assert!(!Charset::Ascii.encode_str("Aé", &mut b));
        assert!(Charset::Ascii.encode_str("A", &mut b) && b == [0x41]);
        let mut s = "x".to_string();
        assert!(Charset::Utf8.decode_into(&[0xC3, 0xA9], &mut s) && s == "é");
        assert!(!Charset::Utf8.decode_into(&[0xC3], &mut s));
        assert!(!Charset::Ascii.decode_into(&[0xC3, 0xA9], &mut s));
        assert!(!Charset::Ascii.decode_into(&[0xE9], &mut s));
        assert!(Charset::Ascii.decode_into(b"A", &mut s) && s == "A");
        assert!(Charset::Latin1.decode_into(&[0xE9], &mut s) && s == "é");
        assert_eq!(Charset::for_name("Latin1"), Some(Charset::Latin1));
        assert_eq!(Charset::for_name("default"), None);
        assert!(boolean_value("ON").unwrap());
        assert!(!boolean_value("off").unwrap());
        assert_eq!(
            pairs("a b,  c d,"),
            Some(vec![("a".into(), "b".into()), ("c".into(), "d".into())])
        );
    }
}
