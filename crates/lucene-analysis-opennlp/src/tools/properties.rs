//! `java.util.Properties.load(InputStream)`: the manifest of a model
//! package. ISO-8859-1 text in logical lines (a line ending in an odd
//! number of backslashes continues on the next, whose leading white space
//! is dropped); `#` and `!` start comment lines; the key ends at the first
//! unescaped `=`, `:` or white space, then white space and one `=`/`:`
//! separate it from the value; `\t \n \r \f \uXXXX` are escapes and any
//! other escaped character stands for itself.

use std::collections::HashMap;

use lucene_analysis::factory::FactoryError;

/// A loaded `Properties`: later keys replace earlier ones, as `put`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Properties {
    map: HashMap<String, String>,
}

fn is_blank(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | 0x0C)
}

impl Properties {
    /// `getProperty(key)`.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }

    /// `setProperty(key, value)`.
    pub fn set(&mut self, key: &str, value: &str) {
        self.map.insert(key.to_string(), value.to_string());
    }

    /// `Properties.load(InputStream)`; a malformed `\uXXXX` is Java's
    /// `IllegalArgumentException`.
    pub fn load(bytes: &[u8]) -> Result<Properties, FactoryError> {
        let mut props = Properties::default();
        for line in logical_lines(bytes) {
            let (key, value) = split_key_value(&line);
            let key = convert(key)?;
            let value = convert(value)?;
            props.map.insert(key, value);
        }
        Ok(props)
    }
}

/// `LineReader.readLine`: the logical lines, comment and blank lines
/// dropped, continuations joined (the trailing backslash removed).
// ARITH: offsets within the input (each read checked).
#[allow(clippy::arithmetic_side_effects)]
fn logical_lines(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut lines = Vec::new();
    let mut i = 0;
    let n = bytes.len();
    let mut current: Vec<u8> = Vec::new();
    let mut continuing = false;
    while i < n {
        // the natural line [i, end)
        let mut end = i;
        while end < n && bytes[end] != b'\n' && bytes[end] != b'\r' {
            end += 1;
        }
        let mut start = i;
        // skip a "\r\n" pair as one terminator
        i = if end < n && bytes[end] == b'\r' && end + 1 < n && bytes[end + 1] == b'\n' {
            end + 2
        } else {
            end + 1
        };
        while start < end && is_blank(bytes[start]) {
            start += 1;
        }
        let natural = &bytes[start..end];
        if !continuing && (natural.is_empty() || natural[0] == b'#' || natural[0] == b'!') {
            continue;
        }
        let backslashes = natural.iter().rev().take_while(|&&c| c == b'\\').count();
        if backslashes % 2 == 1 {
            current.extend_from_slice(&natural[..natural.len() - 1]);
            continuing = true;
            if i >= n {
                // a continuation at end of input ends the line
                lines.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.extend_from_slice(natural);
        continuing = false;
        lines.push(std::mem::take(&mut current));
    }
    lines
}

/// `Properties.load0`: the key (up to an unescaped `=`, `:` or white
/// space) and the value (after the separator).
// ARITH: offsets within the line.
#[allow(clippy::arithmetic_side_effects)]
fn split_key_value(line: &[u8]) -> (&[u8], &[u8]) {
    let mut key_len = 0;
    let mut value_start = line.len();
    let mut has_sep = false;
    let mut preceding_backslash = false;
    while key_len < line.len() {
        let c = line[key_len];
        if (c == b'=' || c == b':') && !preceding_backslash {
            value_start = key_len + 1;
            has_sep = true;
            break;
        } else if is_blank(c) && !preceding_backslash {
            value_start = key_len + 1;
            break;
        }
        preceding_backslash = c == b'\\' && !preceding_backslash;
        key_len += 1;
    }
    while value_start < line.len() {
        let c = line[value_start];
        if !is_blank(c) {
            if !has_sep && (c == b'=' || c == b':') {
                has_sep = true;
            } else {
                break;
            }
        }
        value_start += 1;
    }
    (&line[..key_len], &line[value_start.min(line.len())..])
}

/// `loadConvert`: the escapes resolved, each byte an ISO-8859-1 char.
// ARITH: offsets within the line (each read checked).
#[allow(clippy::arithmetic_side_effects)]
fn convert(s: &[u8]) -> Result<String, FactoryError> {
    let mut units: Vec<u16> = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        i += 1;
        if c != b'\\' {
            units.push(u16::from(c));
            continue;
        }
        let Some(&e) = s.get(i) else {
            // a lone trailing backslash stands for itself
            units.push(u16::from(b'\\'));
            break;
        };
        i += 1;
        match e {
            b'u' => {
                let hex = s
                    .get(i..i + 4)
                    .and_then(|h| std::str::from_utf8(h).ok())
                    .and_then(|h| u16::from_str_radix(h, 16).ok())
                    .filter(|_| s[i..i + 4].iter().all(u8::is_ascii_hexdigit))
                    .ok_or_else(|| FactoryError::illegal_argument("Malformed \\uxxxx encoding."))?;
                units.push(hex);
                i += 4;
            }
            b't' => units.push(u16::from(b'\t')),
            b'r' => units.push(u16::from(b'\r')),
            b'n' => units.push(u16::from(b'\n')),
            b'f' => units.push(0x0C),
            other => units.push(u16::from(other)),
        }
    }
    Ok(String::from_utf16_lossy(&units))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code
    use super::*;

    #[test]
    fn java_properties_syntax() {
        let p = Properties::load(
            b"# comment\n! also\n\n  a=1\nb : 2\nc 3\nd\\=x=4\ne=multi\\\n   line\r\nf=\\u0041\\t\\n\\r\\f\\q\\\\\ng\nh=x\\",
        )
        .unwrap();
        assert_eq!(p.get("a"), Some("1"));
        assert_eq!(p.get("b"), Some("2"));
        assert_eq!(p.get("c"), Some("3"));
        assert_eq!(p.get("d=x"), Some("4"));
        assert_eq!(p.get("e"), Some("multiline"));
        assert_eq!(p.get("f"), Some("A\t\n\r\x0cq\\"));
        assert_eq!(p.get("g"), Some(""));
        assert_eq!(p.get("h"), Some("x"));
        assert_eq!(p.get("missing"), None);
        let p = Properties::load(b"k = = v\nlatin=\xe9\rz:").unwrap();
        assert_eq!(p.get("k"), Some("= v"));
        assert_eq!(p.get("latin"), Some("é"));
        assert_eq!(p.get("z"), Some(""));
        // a continued comment-looking line is not a comment
        let p = Properties::load(b"a=\\\n#b").unwrap();
        assert_eq!(p.get("a"), Some("#b"));
        let p = Properties::load(b"x=a\\").unwrap();
        assert_eq!(p.get("x"), Some("a"));
        let p = Properties::load(b"y=\\\\\\").unwrap();
        assert_eq!(p.get("y"), Some("\\"));
        let mut p = Properties::load(b"k=1\nk=2").unwrap();
        assert_eq!(p.get("k"), Some("2"));
        p.set("k", "3");
        assert_eq!(p.get("k"), Some("3"));
        for bad in [&b"a=\\u00"[..], b"a=\\uZZZZ", b"a=\\u+123"] {
            assert_eq!(
                Properties::load(bad).unwrap_err().message,
                "Malformed \\uxxxx encoding."
            );
        }
    }
}
