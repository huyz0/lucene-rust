//! UTF-16 helpers with Java's `Character`/`UTF16` semantics: an unpaired
//! surrogate is a code point of its own.

/// `UTF16Plus.isLeadSurrogate(c)` / `Character.isHighSurrogate`.
#[inline]
pub fn is_lead(c: i32) -> bool {
    (c & !0x3ff) == 0xd800
}

/// `UTF16Plus.isTrailSurrogate(c)` / `Character.isLowSurrogate`.
#[inline]
pub fn is_trail(c: i32) -> bool {
    (c & !0x3ff) == 0xdc00
}

/// `Character.toCodePoint(lead, trail)`.
#[inline]
pub fn to_code_point(lead: i32, trail: i32) -> i32 {
    // ARITH: lead and trail are surrogates, so the result is < 0x110000.
    #[allow(clippy::arithmetic_side_effects)]
    let c = ((lead - 0xd800) << 10) + (trail - 0xdc00) + 0x10000;
    c
}

/// `Character.codePointAt(s, i)`; `i < s.len()`.
#[inline]
pub fn code_point_at(s: &[u16], i: usize) -> i32 {
    let c = s.get(i).map_or(0, |&u| i32::from(u));
    if is_lead(c) {
        if let Some(&t) = s.get(i.wrapping_add(1)) {
            if is_trail(i32::from(t)) {
                return to_code_point(c, i32::from(t));
            }
        }
    }
    c
}

/// `Character.codePointBefore(s, i)`; `0 < i <= s.len()`.
#[inline]
pub fn code_point_before(s: &[u16], i: usize) -> i32 {
    let c = s.get(i.wrapping_sub(1)).map_or(0, |&u| i32::from(u));
    if is_trail(c) && i >= 2 {
        if let Some(&l) = s.get(i.wrapping_sub(2)) {
            if is_lead(i32::from(l)) {
                return to_code_point(i32::from(l), c);
            }
        }
    }
    c
}

/// The surrogate pair of a supplementary code point.
#[inline]
pub fn surrogates(c: i32) -> (u16, u16) {
    // ARITH: c is a supplementary code point (0x10000..=0x10ffff).
    #[allow(clippy::arithmetic_side_effects)]
    let v = (c - 0x10000) as u32;
    ((0xd800 | (v >> 10)) as u16, (0xdc00 | (v & 0x3ff)) as u16)
}

/// `StringBuilder.appendCodePoint(c)`; a value that is not a code point
/// (only corrupt data produces one) appends U+FFFD where Java throws.
#[inline]
pub fn push_code_point(v: &mut Vec<u16>, c: i32) {
    if (0..=0xffff).contains(&c) {
        v.push(c as u16);
    } else if (0x10000..=0x10ffff).contains(&c) {
        let (l, t) = surrogates(c);
        v.push(l);
        v.push(t);
    } else {
        v.push(0xfffd);
    }
}

/// The UTF-16 units of `s`.
pub fn units(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// `s` as a Rust string (an unpaired surrogate becomes U+FFFD).
pub fn string(s: &[u16]) -> String {
    String::from_utf16_lossy(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_handling() {
        let s = units("a\u{10400}b");
        assert_eq!(code_point_at(&s, 1), 0x10400);
        assert_eq!(code_point_at(&s, 2), 0xdc00);
        assert_eq!(code_point_at(&s, 9), 0);
        assert_eq!(code_point_before(&s, 3), 0x10400);
        assert_eq!(code_point_before(&s, 2), 0xd801);
        assert_eq!(code_point_before(&s, 1), 'a' as i32);
        assert_eq!(code_point_before(&[0xdc00], 1), 0xdc00);
        assert_eq!(code_point_at(&[0xd800], 0), 0xd800);
        let mut v = Vec::new();
        push_code_point(&mut v, 0x10400);
        push_code_point(&mut v, 0x41);
        push_code_point(&mut v, -1);
        push_code_point(&mut v, 0x110000);
        assert_eq!(v, vec![0xd801, 0xdc00, 0x41, 0xfffd, 0xfffd]);
        assert_eq!(string(&v[..3]), "\u{10400}A");
        assert!(is_lead(0xdbff) && !is_lead(0xdc00) && is_trail(0xdfff));
    }
}
