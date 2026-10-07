//! `java.util.Properties.load(Reader)`: the `.info` metadata files'
//! syntax -- `#`/`!` comment lines, `key=value`, `key:value` or
//! `key value`, backslash line continuation, `\uXXXX` and `\t\n\r\f`
//! escapes; a later key replaces an earlier one.

use crate::MorfologikError;

fn is_ws(c: char) -> bool {
    c == ' ' || c == '\t' || c == '\u{c}'
}

/// The logical lines: natural lines joined where one ends in an odd number
/// of backslashes, each continuation's leading whitespace dropped; comment
/// and blank lines skipped.
fn logical_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    let mut rest = text;
    while !rest.is_empty() {
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        let line = &rest[..end];
        rest = &rest[end..];
        if rest.starts_with("\r\n") {
            rest = &rest[2..];
        } else if !rest.is_empty() {
            rest = &rest[1..];
        }
        let trimmed = line.trim_start_matches(is_ws);
        let continued = match current.take() {
            Some(mut c) => {
                c.push_str(trimmed);
                c
            }
            None => {
                if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('!') {
                    continue;
                }
                trimmed.to_string()
            }
        };
        let backslashes = continued.chars().rev().take_while(|&c| c == '\\').count();
        if backslashes % 2 == 1 {
            let mut c = continued;
            c.pop();
            current = Some(c);
        } else {
            out.push(continued);
        }
    }
    if let Some(c) = current {
        out.push(c);
    }
    out
}

/// `Properties.loadConvert`: the escapes.
fn unescape(s: &str) -> Result<String, MorfologikError> {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('u') => {
                let hex: String = it.by_ref().take(4).collect();
                let v = (hex.len() == 4)
                    .then(|| u32::from_str_radix(&hex, 16).ok())
                    .flatten()
                    .ok_or_else(|| {
                        MorfologikError::new(
                            "IllegalArgumentException: Malformed \\uxxxx encoding.",
                        )
                    })?;
                out.push(char::from_u32(v).unwrap_or(char::REPLACEMENT_CHARACTER));
            }
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('f') => out.push('\u{c}'),
            Some(o) => out.push(o),
            None => {}
        }
    }
    Ok(out)
}

/// `Properties.load`: the key/value pairs, a repeated key keeping its last
/// value, in first-appearance order.
pub fn load(text: &str) -> Result<Vec<(String, String)>, MorfologikError> {
    let mut map: Vec<(String, String)> = Vec::new();
    for line in logical_lines(text) {
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        let mut has_sep = false;
        let mut escaped = false;
        while i < chars.len() {
            let c = chars[i];
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '=' || c == ':' {
                has_sep = true;
                break;
            } else if is_ws(c) {
                break;
            }
            // ARITH: i < chars.len().
            #[allow(clippy::arithmetic_side_effects)]
            {
                i += 1;
            }
        }
        let key: String = chars[..i].iter().collect();
        let mut j = i;
        if has_sep {
            // ARITH: chars[i] exists (the separator).
            #[allow(clippy::arithmetic_side_effects)]
            {
                j += 1;
            }
        }
        while j < chars.len() && is_ws(chars[j]) {
            // ARITH: j < chars.len().
            #[allow(clippy::arithmetic_side_effects)]
            {
                j += 1;
            }
        }
        if !has_sep && j < chars.len() && (chars[j] == '=' || chars[j] == ':') {
            // ARITH: j < chars.len().
            #[allow(clippy::arithmetic_side_effects)]
            {
                j += 1;
            }
            while j < chars.len() && is_ws(chars[j]) {
                // ARITH: as above.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    j += 1;
                }
            }
        }
        let value: String = chars[j..].iter().collect();
        let (key, value) = (unescape(&key)?, unescape(&value)?);
        match map.iter_mut().find(|(k, _)| *k == key) {
            Some(e) => e.1 = value,
            None => map.push((key, value)),
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(text: &str) -> Vec<(String, String)> {
        load(text).unwrap()
    }

    #[test]
    fn properties_syntax() {
        let p = kv("# comment\n! also\n\n  a=1\nb : 2\nc 3\nd\\\n   ont=x\\\n  y\r\ne\\=f=g\\u0041\\t\nh\nk=1\nk=2");
        let get = |k: &str| p.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("a"), Some("1"));
        assert_eq!(get("b"), Some("2"));
        assert_eq!(get("c"), Some("3"));
        assert_eq!(get("dont"), Some("xy"));
        assert_eq!(get("e=f"), Some("gA\t"));
        assert_eq!(get("h"), Some(""));
        assert_eq!(get("k"), Some("2"));
        assert_eq!(
            kv("x=\\n\\r\\f\\q"),
            [("x".to_string(), "\n\r\u{c}q".to_string())]
        );
        assert_eq!(kv("x=a\\"), [("x".to_string(), "a".to_string())]);
        assert_eq!(kv("x = = y"), [("x".to_string(), "= y".to_string())]);
        assert!(load("x=\\u12").is_err());
        assert!(load("x=\\uZZZZ").is_err());
        assert_eq!(kv("x=\\\\"), [("x".to_string(), "\\".to_string())]);
    }
}
