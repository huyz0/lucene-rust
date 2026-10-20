//! `SimpleTextUtil`: the line and checksum primitives every SimpleText file
//! is built from, and the number parsing the readers do on each line.
//!
//! A line is bytes up to a `\n`; a `\n` or `\` inside a value is written
//! behind a `\` escape, so a value (a term, a payload) may hold any byte.
//! Every file ends `checksum ` + the CRC32 of everything before that line,
//! zero-padded to 20 decimal digits, + `\n`, and nothing after it.

use super::{corrupt, Result};

/// `SimpleTextUtil.NEWLINE`.
pub const NEWLINE: u8 = 10;
/// `SimpleTextUtil.ESCAPE`.
pub const ESCAPE: u8 = 92;
/// `SimpleTextUtil.CHECKSUM`.
pub const CHECKSUM: &[u8] = b"checksum ";

/// `SimpleTextUtil.write(DataOutput, BytesRef)`: `b`, with each `\n` and `\`
/// behind an escape. (`write(DataOutput, String, BytesRefBuilder)` is this
/// over the string's UTF-8 bytes.)
pub fn write(out: &mut Vec<u8>, b: &[u8]) {
    for &bx in b {
        if bx == NEWLINE || bx == ESCAPE {
            out.push(ESCAPE);
        }
        out.push(bx);
    }
}

/// `SimpleTextUtil.writeNewline`.
pub fn write_newline(out: &mut Vec<u8>) {
    out.push(NEWLINE);
}

/// `SimpleTextUtil.readLine`: the line at `*pos`, unescaped, into `scratch`;
/// `*pos` moves past its `\n`.
///
/// # Errors
/// `Eof` when the input ends before the line's `\n` (Java's `EOFException`).
pub fn read_line(input: &[u8], pos: &mut usize, scratch: &mut Vec<u8>) -> Result<()> {
    scratch.clear();
    loop {
        let b = next_byte(input, pos)?;
        if b == ESCAPE {
            scratch.push(next_byte(input, pos)?);
        } else if b == NEWLINE {
            return Ok(());
        } else {
            scratch.push(b);
        }
    }
}

fn next_byte(input: &[u8], pos: &mut usize) -> Result<u8> {
    let b = *input
        .get(*pos)
        .ok_or(lucene_store::Error::Eof { offset: *pos })?;
    // ARITH: `*pos < input.len()`, just checked by `get`.
    #[allow(clippy::arithmetic_side_effects)]
    {
        *pos += 1;
    }
    Ok(b)
}

/// `String.format(Locale.ROOT, "%020d", checksum)`.
fn format_checksum(checksum: u64) -> String {
    format!("{checksum:020}")
}

/// `SimpleTextUtil.writeChecksum`: the CRC32 of everything `out` holds,
/// as the file's last line.
pub fn write_checksum(out: &mut Vec<u8>) {
    let checksum = format_checksum(u64::from(crc32fast::hash(out)));
    write(out, CHECKSUM);
    write(out, checksum.as_bytes());
    write_newline(out);
}

/// `SimpleTextUtil.checkFooter(ChecksumIndexInput)`, for a checksum input
/// that started reading at `start`: the line at `*pos` must be the CRC32 of
/// `input[start..*pos]`, and the last line of `input`.
///
/// # Errors
/// `Corrupted` with Java's messages: no checksum line, a checksum that
/// differs, bytes after it; `Eof` for a missing line.
pub fn check_footer(input: &[u8], start: usize, pos: &mut usize) -> Result<()> {
    let covered = input
        .get(start..*pos)
        .ok_or_else(|| corrupt("checksum range outside the file"))?;
    let expected = format_checksum(u64::from(crc32fast::hash(covered)));
    let mut scratch = Vec::new();
    read_line(input, pos, &mut scratch)?;
    let Some(actual) = scratch.strip_prefix(CHECKSUM) else {
        return Err(corrupt(format!(
            "SimpleText failure: expected checksum line but got {}",
            String::from_utf8_lossy(&scratch)
        )));
    };
    if actual != expected.as_bytes() {
        return Err(corrupt(format!(
            "SimpleText checksum failure: {} != {expected}",
            String::from_utf8_lossy(actual)
        )));
    }
    if input.len() != *pos {
        return Err(corrupt(
            "Unexpected stuff at the end of file, please be careful with your text editor!",
        ));
    }
    Ok(())
}

/// `SimpleTextUtil.fromBytesRefString`: the inverse of `BytesRef.toString`
/// (`[61 62 ff]`).
///
/// # Errors
/// [`super::Error::IllegalArgument`] for a string not in that shape (Java's
/// `IllegalArgumentException`, and its `NumberFormatException` for a part
/// that is not hex).
pub fn from_bytes_ref_string(s: &str) -> Result<Vec<u8>> {
    let not_bytes_ref = || {
        super::Error::IllegalArgument(format!(
            "string {s} was not created from BytesRef.toString?"
        ))
    };
    let inner = s
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or_else(not_bytes_ref)?;
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    // Java's `split(" ")` drops trailing empty parts ("[61 ]" is [61], "[ ]"
    // empty); then `(byte) Integer.parseInt(part, 16)`, the part's low byte.
    let mut parts: Vec<&str> = inner.split(' ').collect();
    while parts.last() == Some(&"") {
        parts.pop();
    }
    parts
        .into_iter()
        .map(|part| {
            i32::from_str_radix(part, 16)
                .map(|v| v as u8)
                .map_err(|_| super::Error::IllegalArgument(format!("For input string: \"{part}\"")))
        })
        .collect()
}

/// `ArrayUtil.parseInt(chars, 0, len)` over a line's tail: an optional `-`,
/// then decimal digits, within `int`.
///
/// # Errors
/// `Corrupted` for anything else (Java's `NumberFormatException`).
pub fn parse_int(bytes: &[u8]) -> Result<i32> {
    i32::try_from(parse_decimal(
        bytes,
        false,
        i64::from(i32::MIN),
        i64::from(i32::MAX),
    )?)
    .map_err(|_| number_error(bytes))
}

/// `Long.parseLong(String)` over a line's tail: an optional `-` or `+`, then
/// decimal digits, within `long`.
///
/// # Errors
/// `Corrupted` for anything else (Java's `NumberFormatException`).
pub fn parse_long(bytes: &[u8]) -> Result<i64> {
    parse_decimal(bytes, true, i64::MIN, i64::MAX)
}

fn number_error(bytes: &[u8]) -> super::Error {
    corrupt(format!(
        "For input string: \"{}\"",
        String::from_utf8_lossy(bytes)
    ))
}

/// Accumulates negatively, as Java's parsers do, so `MIN` itself parses.
fn parse_decimal(bytes: &[u8], allow_plus: bool, min: i64, max: i64) -> Result<i64> {
    let (negative, digits) = match bytes.split_first() {
        Some((b'-', rest)) => (true, rest),
        Some((b'+', rest)) if allow_plus => (false, rest),
        _ => (false, bytes),
    };
    if digits.is_empty() {
        return Err(number_error(bytes));
    }
    let mut result: i64 = 0;
    for &b in digits {
        let digit = char::from(b)
            .to_digit(10)
            .ok_or_else(|| number_error(bytes))?;
        result = result
            .checked_mul(10)
            .and_then(|r| r.checked_sub(i64::from(digit)))
            .filter(|&r| r >= min)
            .ok_or_else(|| number_error(bytes))?;
    }
    if negative {
        Ok(result)
    } else {
        result
            .checked_neg()
            .filter(|&r| r <= max)
            .ok_or_else(|| number_error(bytes))
    }
}

/// `StringHelper.startsWith(line, prefix)`, and the tail after it.
pub(crate) fn tail<'l>(line: &'l [u8], prefix: &[u8]) -> Option<&'l [u8]> {
    line.strip_prefix(prefix)
}

/// `Integer.parseInt(String)` over a line's tail: an optional `-` or `+`,
/// then decimal digits, within `int`.
///
/// # Errors
/// `Corrupted` for anything else (Java's `NumberFormatException`).
pub fn parse_integer(bytes: &[u8]) -> Result<i32> {
    i32::try_from(parse_decimal(
        bytes,
        true,
        i64::from(i32::MIN),
        i64::from(i32::MAX),
    )?)
    .map_err(|_| number_error(bytes))
}

/// `Boolean.parseBoolean(String)`: `true` ignoring case; anything else is
/// `false`.
pub fn parse_boolean(bytes: &[u8]) -> bool {
    bytes.eq_ignore_ascii_case(b"true")
}

/// `Boolean.toString(boolean)`.
pub fn boolean_str(b: bool) -> &'static [u8] {
    if b {
        b"true"
    } else {
        b"false"
    }
}

/// `prefix`, then `value` escaped, then a newline: the one shape every
/// SimpleText writer's lines take.
pub fn write_line(out: &mut Vec<u8>, prefix: &[u8], value: &[u8]) {
    write(out, prefix);
    write(out, value);
    write_newline(out);
}

/// `SimpleTextUtil.readLine(input, scratch)` then `assert
/// StringHelper.startsWith(scratch, prefix)`: the line's tail after
/// `prefix`.
///
/// # Errors
/// `Eof` for a missing line; `Corrupted` for a line without `prefix` (Java's
/// `assert`, which a production JVM skips).
pub fn read_value<'s>(
    input: &[u8],
    pos: &mut usize,
    scratch: &'s mut Vec<u8>,
    prefix: &[u8],
) -> Result<&'s [u8]> {
    read_line(input, pos, scratch)?;
    tail(scratch, prefix).ok_or_else(|| {
        corrupt(format!(
            "SimpleText failure: expected line starting with {:?} but got {:?}",
            String::from_utf8_lossy(prefix),
            String::from_utf8_lossy(scratch)
        ))
    })
}

/// [`read_value`] as `new String(bytes, UTF_8)` (malformed UTF-8 decodes
/// lossily, as Java's decoder does, though the count of U+FFFD can differ).
///
/// # Errors
/// What [`read_value`] fails with.
pub fn read_string(
    input: &[u8],
    pos: &mut usize,
    scratch: &mut Vec<u8>,
    prefix: &[u8],
) -> Result<String> {
    let v = read_value(input, pos, scratch, prefix)?;
    Ok(String::from_utf8_lossy(v).into_owned())
}
