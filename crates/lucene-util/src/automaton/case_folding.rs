//! `CaseFolding`: the case variants `RegExp`'s case-insensitive flags match.
//!
//! `expand(c)` yields `c`, then `Character.toUpperCase(c)` and
//! `Character.toLowerCase(c)` when they differ from `c`, then Lucene's
//! special-casing extras (Kelvin sign, long s, Greek symbol variants, ...),
//! transcribed from Lucene 10.5.0 (generated there from ICU4J 77.1, Unicode
//! 16.0). The JDK's simple case mappings come from [`super::java_case_table`].

use super::java_case_table::SIMPLE_CASE;

fn lookup(c: i32) -> Option<(u32, u32, u32)> {
    let c = u32::try_from(c).ok()?;
    SIMPLE_CASE
        .binary_search_by_key(&c, |&(k, _, _)| k)
        .ok()
        .map(|i| SIMPLE_CASE[i])
}

/// `Character.toUpperCase(int)`: the JDK's simple uppercase mapping.
pub fn java_to_upper_case(c: i32) -> i32 {
    lookup(c).map_or(c, |(_, u, _)| u as i32)
}

/// `Character.toLowerCase(int)`: the JDK's simple lowercase mapping.
pub fn java_to_lower_case(c: i32) -> i32 {
    lookup(c).map_or(c, |(_, _, l)| l as i32)
}

/// Lucene's hand-listed extra variants beyond the simple case mappings.
fn special(c: i32) -> &'static [i32] {
    match c {
        0x004B => &[0x212A],
        0x0053 => &[0x017F],
        0x006B => &[0x212A],
        0x0073 => &[0x017F],
        0x00B5 => &[0x03BC],
        0x00C5 => &[0x212B],
        0x00DF => &[0x1E9E],
        0x00E5 => &[0x212B],
        0x017F => &[0x0073],
        0x01C4 => &[0x01C5],
        0x01C6 => &[0x01C5],
        0x01C7 => &[0x01C8],
        0x01C9 => &[0x01C8],
        0x01CA => &[0x01CB],
        0x01CC => &[0x01CB],
        0x01F1 => &[0x01F2],
        0x01F3 => &[0x01F2],
        0x0345 => &[0x03B9, 0x1FBE],
        0x0390 => &[0x1FD3],
        0x0392 => &[0x03D0],
        0x0395 => &[0x03F5],
        0x0398 => &[0x03D1, 0x03F4],
        0x0399 => &[0x0345, 0x1FBE],
        0x039A => &[0x03F0],
        0x039C => &[0x00B5],
        0x03A0 => &[0x03D6],
        0x03A1 => &[0x03F1],
        0x03A3 => &[0x03C2],
        0x03A6 => &[0x03D5],
        0x03A9 => &[0x2126],
        0x03B0 => &[0x1FE3],
        0x03B2 => &[0x03D0],
        0x03B5 => &[0x03F5],
        0x03B8 => &[0x03D1, 0x03F4],
        0x03B9 => &[0x0345, 0x1FBE],
        0x03BA => &[0x03F0],
        0x03BC => &[0x00B5],
        0x03C0 => &[0x03D6],
        0x03C1 => &[0x03F1],
        0x03C2 => &[0x03C3],
        0x03C3 => &[0x03C2],
        0x03C6 => &[0x03D5],
        0x03C9 => &[0x2126],
        0x03D0 => &[0x03B2],
        0x03D1 => &[0x03B8, 0x03F4],
        0x03D5 => &[0x03C6],
        0x03D6 => &[0x03C0],
        0x03F0 => &[0x03BA],
        0x03F1 => &[0x03C1],
        0x03F4 => &[0x0398, 0x03D1],
        0x03F5 => &[0x03B5],
        0x0412 => &[0x1C80],
        0x0414 => &[0x1C81],
        0x041E => &[0x1C82],
        0x0421 => &[0x1C83],
        0x0422 => &[0x1C84, 0x1C85],
        0x042A => &[0x1C86],
        0x0432 => &[0x1C80],
        0x0434 => &[0x1C81],
        0x043E => &[0x1C82],
        0x0441 => &[0x1C83],
        0x0442 => &[0x1C84, 0x1C85],
        0x044A => &[0x1C86],
        0x0462 => &[0x1C87],
        0x0463 => &[0x1C87],
        0x1C80 => &[0x0432],
        0x1C81 => &[0x0434],
        0x1C82 => &[0x043E],
        0x1C83 => &[0x0441],
        0x1C84 => &[0x0442, 0x1C85],
        0x1C85 => &[0x0442, 0x1C84],
        0x1C86 => &[0x044A],
        0x1C87 => &[0x0463],
        0x1C88 => &[0xA64B],
        0x1E60 => &[0x1E9B],
        0x1E61 => &[0x1E9B],
        0x1E9B => &[0x1E61],
        0x1FBE => &[0x0345, 0x03B9],
        0x1FD3 => &[0x0390],
        0x1FE3 => &[0x03B0],
        0x2126 => &[0x03A9],
        0x212A => &[0x004B],
        0x212B => &[0x00C5],
        0xA64A => &[0x1C88],
        0xA64B => &[0x1C88],
        0xFB05 => &[0xFB06],
        0xFB06 => &[0xFB05],
        _ => &[],
    }
}

/// `CaseFolding.expand(c, fn)`: call `f` with `c` and each case variant.
pub fn expand(c: i32, f: &mut dyn FnMut(i32)) {
    f(c);
    let upper = java_to_upper_case(c);
    if upper != c {
        f(upper);
    }
    let lower = java_to_lower_case(c);
    if lower != c {
        f(lower);
    }
    for &v in special(c) {
        f(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(c: i32) -> Vec<i32> {
        let mut v = Vec::new();
        expand(c, &mut |x| v.push(x));
        v
    }

    #[test]
    fn expansions() {
        assert_eq!(all('a' as i32), vec![97, 65]);
        assert_eq!(all('k' as i32), vec![107, 75, 0x212A]);
        assert_eq!(all(0x3A3), vec![0x3A3, 0x3C3, 0x3C2]);
        assert_eq!(all('1' as i32), vec![49]);
        assert_eq!(all(-5), vec![-5]);
        // Simple, not full, mappings: sharp s has no simple uppercase.
        assert_eq!(java_to_upper_case(0xDF), 0xDF);
        assert_eq!(java_to_lower_case(0x130), 'i' as i32);
        assert_eq!(java_to_upper_case(0x1C5), 0x1C4);
        assert_eq!(java_to_lower_case(0x1C5), 0x1C6);
    }
}
