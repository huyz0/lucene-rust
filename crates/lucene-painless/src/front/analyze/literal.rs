//! Numeric literals: their type and value, as Painless reads them.

use super::{compile_error, Res};
use crate::front::ast::NumKind;
use crate::ir::Constant;

/// A literal's constant and Painless type. `negative` is a literal under a
/// unary minus, which reads `-2147483648` as an `int`.
pub fn number(
    pos: u32,
    kind: NumKind,
    text: &str,
    negative: bool,
) -> Res<(Constant, &'static str)> {
    let bad = |what: &str| compile_error(pos, format!("invalid {what} literal [{text}]"));
    let sign = if negative { "-" } else { "" };
    let last = text.bytes().last().unwrap_or(b'0');
    match kind {
        NumKind::Octal | NumKind::Hex | NumKind::Integer => {
            let (digits, radix) = match kind {
                NumKind::Hex => (&text[2..], 16),
                NumKind::Octal => (&text[1..], 8),
                _ => (text, 10),
            };
            match last {
                b'l' | b'L' => {
                    let d = &digits[..digits.len() - 1];
                    i64::from_str_radix(&format!("{sign}{d}"), radix)
                        .map(|v| (Constant::Long(v), "long"))
                        .or_else(|_| bad("long"))
                }
                b'f' | b'F' | b'd' | b'D' if kind == NumKind::Integer => {
                    let d = &digits[..digits.len() - 1];
                    decimal(pos, &format!("{sign}{d}"), last)
                }
                _ => i32::from_str_radix(&format!("{sign}{digits}"), radix)
                    .map(|v| (Constant::Int(v), "int"))
                    .or_else(|_| bad("int")),
            }
        }
        NumKind::Decimal => {
            let (d, sfx) = if matches!(last, b'f' | b'F' | b'd' | b'D') {
                (&text[..text.len() - 1], last)
            } else {
                (text, b'd')
            };
            decimal(pos, &format!("{sign}{d}"), sfx)
        }
    }
}

fn decimal(pos: u32, text: &str, suffix: u8) -> Res<(Constant, &'static str)> {
    let bad = || compile_error(pos, format!("invalid number literal [{text}]"));
    match suffix {
        b'f' | b'F' => text
            .parse::<f32>()
            .map(|v| (Constant::Float(v), "float"))
            .or_else(|_| bad()),
        _ => text
            .parse::<f64>()
            .map(|v| (Constant::Double(v), "double"))
            .or_else(|_| bad()),
    }
}
