//! What every tool shares: where it writes, how it fails, and the two Java
//! number formats the tools print with.
//!
//! A tool is `fn(args, io) -> ToolResult`: `args` are its own arguments (the
//! Java `main`'s `String[] args`), `io` its `System.out`/`System.err`. It
//! returns the exit code `System.exit` would have set, or the exception Java
//! would have thrown out of `main` -- which [`crate::run`] prints as the JVM
//! does and turns into exit code 1.

use std::io::Write;

/// An exception thrown out of a tool's `main`.
pub type Error = Box<dyn std::error::Error>;

/// A tool's exit code, or the exception that ended it.
pub type ToolResult = Result<i32, Error>;

/// A tool's `System.out` and `System.err`.
pub struct Io<'a> {
    pub out: &'a mut dyn Write,
    pub err: &'a mut dyn Write,
}

/// A Java tool's entry point: `main(String[] args)`.
pub type Tool = fn(&[String], &mut Io<'_>) -> ToolResult;

/// An exception with only a message, as `new Exception(message)` or
/// `new RuntimeException(message)`.
#[derive(Debug)]
pub struct Exception(pub String);

impl std::fmt::Display for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Exception {}

/// `Exception(message)` boxed as a tool's error.
pub fn exception(message: impl Into<String>) -> Error {
    Box::new(Exception(message.into()))
}

/// `String.format(Locale.ROOT, "%,d", n)` (and `DecimalFormat("###,###")`):
/// the digits grouped by three with `,`.
pub fn group_digits(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::with_capacity(digits.len().saturating_mul(2));
    if n < 0 {
        out.push('-');
    }
    let len = digits.len();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (len.saturating_sub(i)) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `Integer.parseInt(s)`: an optional sign and ASCII digits, in `int` range.
///
/// # Errors
/// `NumberFormatException`'s message, `For input string: "s"`.
pub fn parse_int(s: &str) -> Result<i32, String> {
    // `i32::from_str` takes the same `[+-]?digits`, and rejects the empty
    // string, a lone sign and an out-of-range value, as Java does.
    s.parse::<i32>()
        .map_err(|_| format!("For input string: \"{s}\""))
}

/// `args[i]`, or the `ArrayIndexOutOfBoundsException` Java's `args[++i]`
/// throws past the end.
pub fn arg(args: &[String], i: usize) -> Result<&str, Error> {
    args.get(i).map(String::as_str).ok_or_else(|| {
        exception(format!(
            "java.lang.ArrayIndexOutOfBoundsException: Index {i} out of bounds for length {}",
            args.len()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_group_by_three() {
        assert_eq!(group_digits(0), "0");
        assert_eq!(group_digits(999), "999");
        assert_eq!(group_digits(1000), "1,000");
        assert_eq!(group_digits(1234567), "1,234,567");
        assert_eq!(group_digits(-1234), "-1,234");
        assert_eq!(group_digits(i64::MIN), "-9,223,372,036,854,775,808");
    }

    #[test]
    fn parse_int_is_integer_parse_int() {
        assert_eq!(parse_int("42"), Ok(42));
        assert_eq!(parse_int("+7"), Ok(7));
        assert_eq!(parse_int("-3"), Ok(-3));
        assert_eq!(parse_int("x").unwrap_err(), "For input string: \"x\"");
        assert!(parse_int("").is_err());
        assert!(parse_int("-").is_err());
        assert!(parse_int("2147483648").is_err());
    }

    #[test]
    fn arg_past_the_end_is_javas_exception() {
        let args = vec!["-x".to_string()];
        assert_eq!(arg(&args, 0).unwrap(), "-x");
        assert_eq!(
            arg(&args, 1).unwrap_err().to_string(),
            "java.lang.ArrayIndexOutOfBoundsException: Index 1 out of bounds for length 1"
        );
        assert_eq!(exception("m").to_string(), "m");
    }
}
