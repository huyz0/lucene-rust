//! The template and variables of one message, and their stored bytes.
//!
//! - The template is the message with each variable token replaced by
//!   [`VARIABLE`] (U+0002). A token that itself holds U+0002 is always a
//!   variable, so the template's U+0002 characters are exactly its slots.
//! - The variables are stored as, for each slot in order, a vInt byte length
//!   and the token's UTF-8 bytes ([`encode_args`]).

use crate::{Error, Result};

/// The character that stands for one variable in a template.
pub const VARIABLE: char = '\u{2}';

/// A message split into its template and variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split<'a> {
    /// The message with each variable token replaced by [`VARIABLE`].
    pub template: String,
    /// The variable tokens, in message order.
    pub args: Vec<&'a str>,
}

fn is_variable(token: &str) -> bool {
    token.bytes().any(|b| b.is_ascii_digit()) || token.contains(VARIABLE)
}

/// `message` split on single spaces into its template and variables. Empty
/// tokens (runs of spaces, leading or trailing spaces) stay in the template,
/// so [`join`] rebuilds `message` exactly.
pub fn split(message: &str) -> Split<'_> {
    let mut template = String::with_capacity(message.len());
    let mut args = Vec::new();
    for (i, token) in message.split(' ').enumerate() {
        if i > 0 {
            template.push(' ');
        }
        if is_variable(token) {
            template.push(VARIABLE);
            args.push(token);
        } else {
            template.push_str(token);
        }
    }
    Split { template, args }
}

/// The message `template` and `args` were split from.
///
/// # Errors
/// [`Error::Corrupt`] when the number of variables is not the template's
/// number of slots.
pub fn join<S: AsRef<str>>(template: &str, args: &[S]) -> Result<String> {
    let extra: usize = args.iter().map(|a| a.as_ref().len()).sum();
    let mut out = String::with_capacity(template.len().saturating_add(extra));
    let mut next = args.iter();
    for c in template.chars() {
        if c == VARIABLE {
            let a = next
                .next()
                .ok_or_else(|| Error::Corrupt("fewer variables than template slots".into()))?;
            out.push_str(a.as_ref());
        } else {
            out.push(c);
        }
    }
    if next.next().is_some() {
        return Err(Error::Corrupt("more variables than template slots".into()));
    }
    Ok(out)
}

/// The stored bytes of `args`: per variable, a vInt length and its bytes.
pub fn encode_args<S: AsRef<str>>(args: &[S]) -> Vec<u8> {
    let total: usize = args
        .iter()
        .map(|a| a.as_ref().len().saturating_add(5))
        .sum();
    let mut out = Vec::with_capacity(total);
    for a in args {
        let bytes = a.as_ref().as_bytes();
        let mut n = bytes.len();
        while n >= 0x80 {
            // ARITH: the low seven bits of `n`, with the continuation bit.
            #[allow(clippy::cast_possible_truncation)]
            out.push((n & 0x7f) as u8 | 0x80);
            n >>= 7;
        }
        // ARITH: `n < 0x80` here.
        #[allow(clippy::cast_possible_truncation)]
        out.push(n as u8);
        out.extend_from_slice(bytes);
    }
    out
}

/// The variables [`encode_args`] wrote into `bytes`, appended to `out`.
///
/// # Errors
/// [`Error::Corrupt`] for a truncated length or value, or a value that is
/// not UTF-8.
pub fn decode_args<'a>(bytes: &'a [u8], out: &mut Vec<&'a str>) -> Result<()> {
    let corrupt = |m: &str| Error::Corrupt(m.to_string());
    let mut pos = 0usize;
    while pos < bytes.len() {
        let mut len = 0usize;
        let mut shift = 0u32;
        loop {
            let b = *bytes.get(pos).ok_or_else(|| corrupt("truncated length"))?;
            pos = pos.saturating_add(1);
            if shift > 28 {
                return Err(corrupt("length too long"));
            }
            len |= usize::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift = shift.saturating_add(7);
        }
        let end = pos
            .checked_add(len)
            .ok_or_else(|| corrupt("length overflow"))?;
        let value = bytes
            .get(pos..end)
            .ok_or_else(|| corrupt("truncated value"))?;
        out.push(std::str::from_utf8(value).map_err(|_| corrupt("variable is not UTF-8"))?);
        pos = end;
    }
    Ok(())
}

/// The number of variable slots in `template`.
pub fn slots(template: &str) -> usize {
    template.chars().filter(|&c| c == VARIABLE).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(m: &str) -> Split<'_> {
        let s = split(m);
        assert_eq!(join(&s.template, &s.args).unwrap(), m, "{m:?}");
        let bytes = encode_args(&s.args);
        let mut back = Vec::new();
        decode_args(&bytes, &mut back).unwrap();
        assert_eq!(back, s.args);
        assert_eq!(slots(&s.template), s.args.len());
        s
    }

    #[test]
    fn digits_make_variables() {
        let s = roundtrip("081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /10.250.19.102:54106");
        assert_eq!(
            s.template,
            "\u{2} \u{2} \u{2} INFO dfs.DataNode$DataXceiver: Receiving block \u{2} src: \u{2}"
        );
        assert_eq!(
            s.args,
            ["081109", "203518", "143", "blk_-16", "/10.250.19.102:54106"]
        );
    }

    #[test]
    fn spacing_and_edge_cases_rebuild_exactly() {
        for m in [
            "",
            " ",
            "  two  spaces ",
            "no variables here",
            "1",
            "tab\tseparated 42\tvalues",
            "unicode ünïcødé 3ms",
            "a \u{2} literal slot char",
            "x\u{1}y 7",
        ] {
            roundtrip(m);
        }
        assert_eq!(split("a \u{2} b").args, ["\u{2}"]);
    }

    #[test]
    fn long_variables_use_multi_byte_lengths() {
        let long = "9".repeat(300);
        let m = format!("id {long} done");
        let s = roundtrip(&m);
        assert_eq!(s.args, [long.as_str()]);
    }

    #[test]
    fn corrupt_values_are_refused() {
        assert!(join("\u{2} \u{2}", &["a"]).is_err());
        assert!(join("x", &["a"]).is_err());
        let mut out = Vec::new();
        assert!(decode_args(&[5, b'a'], &mut out).is_err());
        assert!(decode_args(&[0x80], &mut out).is_err());
        assert!(decode_args(&[0xff, 0xff, 0xff, 0xff, 0xff, 0x01], &mut out).is_err());
        assert!(decode_args(&[1, 0xff], &mut out).is_err());
    }
}
