//! JSON as ml-commons' libraries see it: a tree that keeps each number's
//! text (so a value read from a remote response goes back out as the remote
//! wrote it) and an object's key order, and the string escapes of Gson and
//! of Apache Commons Text's `escapeJson`, which ml-commons applies to
//! parameters.

use std::fmt::Write as _;

/// A JSON value. Numbers keep their source text; objects their key order.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, as written.
    Number(String),
    /// A string, unescaped.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, in key order; a repeated key keeps its last value.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parses one JSON value (surrounding whitespace allowed).
    pub fn parse(text: &str) -> Option<Json> {
        Self::parse_with(text, false)
    }

    /// Parses one value as ml-commons reads a parameter or script output,
    /// through both org.json and lenient Gson: also unquoted words (`hi`,
    /// `NaN`, keys), single-quoted strings, `;` between object members, and
    /// empty or trailing array elements (read as `null`, dropped at the end).
    pub fn parse_lenient(text: &str) -> Option<Json> {
        Self::parse_with(text, true)
    }

    fn parse_with(text: &str, lenient: bool) -> Option<Json> {
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
            lenient,
        };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        (p.i == p.s.len()).then_some(v)
    }

    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(m) => m.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Whether this is an object or an array, the only values ml-commons'
    /// `isJson` accepts.
    pub fn is_container(&self) -> bool {
        matches!(self, Json::Object(_) | Json::Array(_))
    }

    /// The value as a number, when it is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => n.parse().ok(),
            _ => None,
        }
    }

    /// Whether this number is integral as JSON parsers type it (no fraction
    /// or exponent), as Jackson and json-smart read it into an `Integer` or
    /// `Long`.
    pub fn is_integral(&self) -> bool {
        matches!(self, Json::Number(n) if !n.contains(['.', 'e', 'E']))
    }

    /// Compact JSON, numbers as written, strings escaped as Gson does with
    /// HTML escaping off.
    pub fn to_compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, &|n, out| out.push_str(n));
        out
    }

    /// Compact JSON as Gson writes a value it read itself into a `Map`:
    /// every number became a `Double`, printed by `Double.toString`.
    pub fn to_gson_doubles(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, &|n, out| match n.parse::<f64>() {
            Ok(v) if v.is_finite() => out.push_str(&lucene_util::geo::java_double_string(v)),
            _ => out.push_str(n),
        });
        out
    }

    fn write(&self, out: &mut String, number: &dyn Fn(&str, &mut String)) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Number(n) => number(n, out),
            Json::String(s) => gson_quote(s, out),
            Json::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out, number);
                }
                out.push(']');
            }
            Json::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    gson_quote(k, out);
                    out.push(':');
                    v.write(out, number);
                }
                out.push('}');
            }
        }
    }
}

/// `s` as a JSON string literal, as Gson writes it with HTML escaping off:
/// quote, backslash and control characters escaped, and the line and
/// paragraph separators, which JavaScript does not allow raw.
pub fn gson_quote(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\u{c}' => out.push_str("\\f"),
            '\u{2028}' | '\u{2029}' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `s` quoted by Gson with the quotes taken off: ml-commons' `processTextDoc`.
// ARITH: lengths of an in-memory string, which is at least the two quotes
// `gson_quote` wrote.
#[allow(clippy::arithmetic_side_effects)]
pub fn gson_inner(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    gson_quote(s, &mut out);
    out[1..out.len() - 1].to_string()
}

/// Apache Commons Text `StringEscapeUtils.escapeJson`: quote, backslash and
/// slash escaped, control characters as `\b \f \n \r \t` or `\u00XX`, and
/// every character above U+007F as `\uXXXX` UTF-16 units (upper-case hex).
// ARITH: a capacity a few bytes over an in-memory string's length.
#[allow(clippy::arithmetic_side_effects)]
pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '/' => out.push_str("\\/"),
            '\u{8}' => out.push_str("\\b"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (' '..='\u{7f}').contains(&c) => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{:04X}", u);
                }
            }
        }
    }
    out
}

/// Whether `s` is JSON as ml-commons' `isJson` decides: an object or an
/// array.
pub fn is_json(s: &str) -> bool {
    !s.trim().is_empty() && Json::parse_lenient(s).is_some_and(|v| v.is_container())
}

/// Whether `s` is JSON, or newline-delimited JSON whose every non-empty line
/// is.
pub fn is_json_or_ndjson(s: &str) -> bool {
    if s.trim().is_empty() {
        return false;
    }
    is_json(s)
        || s.split('\n')
            .map(|l| l.strip_suffix('\r').unwrap_or(l).trim())
            .all(|l| l.is_empty() || is_json(l))
}

/// Nesting deeper than this is refused rather than recursed into.
const MAX_DEPTH: usize = 512;

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    lenient: bool,
}

// ARITH: `i` only moves forward over bytes already seen in `s` (each step
// follows a bounds-checked read or `starts_with`), so it never passes
// `s.len()`; `depth` is refused past MAX_DEPTH; a surrogate pair is combined
// only after both halves were range-checked.
#[allow(clippy::arithmetic_side_effects)]
impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.s.get(self.i) == Some(&b) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn lit(&mut self, word: &[u8], v: Json) -> Option<Json> {
        self.s[self.i..].starts_with(word).then(|| {
            self.i += word.len();
            v
        })
    }

    fn value(&mut self, depth: usize) -> Option<Json> {
        if depth > MAX_DEPTH {
            return None;
        }
        if self.lenient {
            match *self.s.get(self.i)? {
                b'"' | b'\'' => return self.string().map(Json::String),
                b'[' | b'{' => {}
                _ => return self.word(),
            }
        }
        match *self.s.get(self.i)? {
            b'n' => self.lit(b"null", Json::Null),
            b't' => self.lit(b"true", Json::Bool(true)),
            b'f' => self.lit(b"false", Json::Bool(false)),
            b'"' => self.string().map(Json::String),
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.eat(b']') {
                    return Some(Json::Array(a));
                }
                loop {
                    self.ws();
                    if self.lenient && matches!(self.s.get(self.i), Some(b',' | b']')) {
                        // An empty element: null, or nothing before `]`.
                        if self.eat(b']') {
                            return Some(Json::Array(a));
                        }
                        a.push(Json::Null);
                    } else {
                        a.push(self.value(depth + 1)?);
                    }
                    self.ws();
                    if self.eat(b']') {
                        return Some(Json::Array(a));
                    }
                    if !self.eat(b',') {
                        return None;
                    }
                }
            }
            b'{' => {
                self.i += 1;
                let mut m: Vec<(String, Json)> = Vec::new();
                self.ws();
                if self.eat(b'}') {
                    return Some(Json::Object(m));
                }
                loop {
                    self.ws();
                    let k = if self.lenient {
                        match self.value(depth + 1)? {
                            Json::String(k) | Json::Number(k) => k,
                            Json::Bool(b) => b.to_string(),
                            Json::Null => "null".into(),
                            _ => return None,
                        }
                    } else {
                        self.string()?
                    };
                    self.ws();
                    if !self.eat(b':') {
                        return None;
                    }
                    self.ws();
                    let v = self.value(depth + 1)?;
                    match m.iter_mut().find(|(key, _)| *key == k) {
                        Some(slot) => slot.1 = v,
                        None => m.push((k, v)),
                    }
                    self.ws();
                    if self.eat(b'}') {
                        return Some(Json::Object(m));
                    }
                    if !self.eat(b',') && !(self.lenient && self.eat(b';')) {
                        return None;
                    }
                }
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    /// An unquoted word: a keyword in any case, a number, or else a string.
    fn word(&mut self) -> Option<Json> {
        let start = self.i;
        while self.i < self.s.len()
            && !matches!(
                self.s[self.i],
                b' ' | b'\t'
                    | b'\n'
                    | b'\r'
                    | b'\x0c'
                    | b','
                    | b':'
                    | b']'
                    | b'}'
                    | b'['
                    | b'{'
                    | b'/'
                    | b'\\'
                    | b'"'
                    | b';'
                    | b'='
                    | b'#'
            )
            && self.s[self.i] >= 0x20
        {
            self.i += 1;
        }
        let w = std::str::from_utf8(&self.s[start..self.i]).ok()?;
        if w.is_empty() {
            return None;
        }
        Some(match w.to_ascii_lowercase().as_str() {
            "null" => Json::Null,
            "true" => Json::Bool(true),
            "false" => Json::Bool(false),
            _ => {
                let mut strict = Parser {
                    s: w.as_bytes(),
                    i: 0,
                    lenient: false,
                };
                match strict.number() {
                    Some(n) if strict.i == w.len() => n,
                    _ => Json::String(w.to_string()),
                }
            }
        })
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        self.eat(b'-');
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i > s
        };
        if self.eat(b'0') {
        } else if !digits(self) {
            return None;
        }
        if self.eat(b'.') && !digits(self) {
            return None;
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if !self.eat(b'+') {
                self.eat(b'-');
            }
            if !digits(self) {
                return None;
            }
        }
        Some(Json::Number(
            std::str::from_utf8(&self.s[start..self.i])
                .ok()?
                .to_string(),
        ))
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = std::str::from_utf8(self.s.get(self.i..self.i + 4)?).ok()?;
        self.i += 4;
        u32::from_str_radix(h, 16).ok()
    }

    fn string(&mut self) -> Option<String> {
        let quote = if self.eat(b'"') {
            b'"'
        } else if self.lenient && self.eat(b'\'') {
            b'\''
        } else {
            return None;
        };
        let mut out = Vec::new();
        loop {
            let b = *self.s.get(self.i)?;
            self.i += 1;
            match b {
                b if b == quote => return String::from_utf8(out).ok(),
                b'\\' => {
                    let e = *self.s.get(self.i)?;
                    self.i += 1;
                    let c = match e {
                        b'"' => '"',
                        b'\'' if self.lenient => '\'',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let u = self.hex4()?;
                            if (0xD800..0xDC00).contains(&u) && self.s[self.i..].starts_with(b"\\u")
                            {
                                self.i += 2;
                                let l = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&l) {
                                    return None;
                                }
                                char::from_u32(0x10000 + ((u - 0xD800) << 10) + (l - 0xDC00))?
                            } else {
                                char::from_u32(u).unwrap_or('\u{fffd}')
                            }
                        }
                        _ => return None,
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                b if b < 0x20 => return None,
                b => out.push(b),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_numbers_and_order() {
        let v = Json::parse(r#" {"b": [1, 2.50, -3e-5], "a": {"x": null, "y": true}, "b": "s"} "#)
            .unwrap();
        assert_eq!(v.to_compact(), r#"{"b":"s","a":{"x":null,"y":true}}"#);
        let v = Json::parse("[1, 2.50, -3e-5, 1e300]").unwrap();
        assert_eq!(v.to_compact(), "[1,2.50,-3e-5,1e300]");
        assert_eq!(v.to_gson_doubles(), "[1.0,2.5,-3.0E-5,1.0E300]");
        assert!(Json::Array(vec![Json::Number("1".into())])
            .get("x")
            .is_none());
        assert!(v.get("x").is_none());
        let Json::Array(a) = &v else { panic!() };
        assert!(a[0].is_integral() && !a[1].is_integral());
        assert_eq!(a[1].as_f64(), Some(2.5));
        for bad in [
            "",
            "[1,]",
            "{\"a\" 1}",
            "01",
            "1.",
            "\"\u{1}\"",
            "nul",
            "[1] x",
            "\"\\x\"",
        ] {
            assert!(Json::parse(bad).is_none(), "{bad:?}");
        }
        assert_eq!(
            Json::parse(r#""\ud83d\ude00\u00e9\n\/""#).unwrap(),
            Json::String("😀é\n/".into())
        );
        let deep = "[".repeat(MAX_DEPTH + 2);
        assert!(Json::parse(&deep).is_none());
    }

    #[test]
    fn escapes_as_gson_and_commons_text() {
        let s = "a\"b\\c/d\n\t\u{1}é😀\u{2028}<>";
        assert_eq!(gson_inner(s), "a\\\"b\\\\c/d\\n\\t\\u0001é😀\\u2028<>");
        assert_eq!(
            escape_json(s),
            "a\\\"b\\\\c\\/d\\n\\t\\u0001\\u00E9\\uD83D\\uDE00\\u2028<>"
        );
        assert_eq!(
            escape_json("\u{7f}\r\u{8}\u{c}\u{80}"),
            "\u{7f}\\r\\b\\f\\u0080"
        );
    }

    #[test]
    fn tells_json_from_text() {
        assert!(is_json("{\"a\":1}") && is_json(" [1] "));
        assert!(!is_json("1") && !is_json("\"s\"") && !is_json("  ") && !is_json("{"));
        assert!(is_json_or_ndjson("{\"a\":1}\n{\"b\":2}\r\n\n"));
        // As stock accepted or refused each as a request payload.
        for ok in [
            "{\"q\": hi}",
            "{\"q\": 's'}",
            "[1,,2]",
            "[1,]",
            "{a:1;b:2}",
            "[NaN]",
            "{\"q\": hé}",
            "{a:'b'}",
        ] {
            assert!(is_json(ok), "{ok:?}");
        }
        for bad in [
            "{\"q\": hi there}",
            "[/*c*/1]",
            "{\"a\":1,}",
            "[1 2]",
            "{\"a\"=1}",
            "{\"a\"=>1}",
            "[#x\n1]",
            "{\"q\": 1} x",
            "[a/b]",
            "[a\\b]",
            "[1;2]",
            "[\"x\"y]",
        ] {
            assert!(!is_json(bad), "{bad:?}");
        }
        assert_eq!(
            Json::parse_lenient("{a:'b\\'c', 'k':[TRUE,,1e2,x1,]}").unwrap(),
            Json::Object(vec![
                ("a".into(), Json::String("b'c".into())),
                (
                    "k".into(),
                    Json::Array(vec![
                        Json::Bool(true),
                        Json::Null,
                        Json::Number("1e2".into()),
                        Json::String("x1".into())
                    ])
                ),
            ])
        );
        assert!(!is_json_or_ndjson("{\"a\":1}\nx") && !is_json_or_ndjson(""));
    }
}
