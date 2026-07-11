//! Java's `Long.toString(n, Character.MAX_RADIX)` / `Long.parseLong(s, 36)` —
//! used for the generation suffix in `segments_N` file names and the matching
//! index-header suffix inside the file (see `SegmentInfos`, `IndexFileNames`).

/// Formats `n` in base 36, lowercase digits, matching `Long.toString(n, 36)`.
pub fn to_base36(n: i64) -> String {
    if n == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let negative = n < 0;
    let mut buf = Vec::new();
    let mut n = n.unsigned_abs();
    while n > 0 {
        buf.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    if negative {
        buf.push(b'-');
    }
    buf.reverse();
    String::from_utf8(buf).unwrap()
}

/// Parses a base-36 string as `Long.parseLong(s, 36)` would (no sign handling
/// beyond what `i64::from_str_radix` already provides).
pub fn from_base36(s: &str) -> Option<i64> {
    i64::from_str_radix(s, 36).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        for n in [0i64, 1, 2, 35, 36, 1000, i64::MAX, i64::MIN + 1] {
            assert_eq!(from_base36(&to_base36(n)), Some(n), "n={n}");
        }
    }
}
