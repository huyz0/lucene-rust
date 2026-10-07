//! `org.apache.lucene.analysis.util.StemmerUtil`: the `char[]`/length
//! helpers the light and minimal stemmers share, over UTF-16 units.

/// `startsWith(char[] s, int len, String prefix)`.
#[inline]
pub fn starts_with(s: &[u16], len: usize, prefix: &str) -> bool {
    for (i, p) in prefix.encode_utf16().enumerate() {
        if i >= len || s[i] != p {
            return false;
        }
    }
    true
}

/// `endsWith(char[] s, int len, String suffix)`.
///
/// Compared from the end in UTF-8, each term unit encoded on the fly against
/// the literal's bytes, so the usual miss costs one unit; a surrogate in the
/// term takes the UTF-16 path.
#[inline]
pub fn ends_with(s: &[u16], len: usize, suffix: &str) -> bool {
    let b = suffix.as_bytes();
    let (mut i, mut j) = (len, b.len());
    while j > 0 {
        if i == 0 {
            return false;
        }
        let u = s[i - 1];
        // Last byte first: that is where nearly every miss shows.
        let last = if u < 0x80 {
            u as u8
        } else {
            0x80 | (u & 0x3F) as u8
        };
        if b[j - 1] != last {
            return false;
        }
        let n = match u {
            0..=0x7F => 1,
            0x80..=0x7FF => {
                if j < 2 || b[j - 2] != 0xC0 | (u >> 6) as u8 {
                    return false;
                }
                2
            }
            0xD800..=0xDFFF => return ends_with_slow(s, len, suffix),
            _ => {
                if j < 3
                    || b[j - 2] != 0x80 | ((u >> 6) & 0x3F) as u8
                    || b[j - 3] != 0xE0 | (u >> 12) as u8
                {
                    return false;
                }
                3
            }
        };
        j -= n;
        i -= 1;
    }
    true
}

/// [`ends_with`] over the literal's UTF-16 units.
fn ends_with_slow(s: &[u16], len: usize, suffix: &str) -> bool {
    let mut i = len;
    for c in suffix.chars().rev() {
        let mut buf = [0u16; 2];
        for &u in c.encode_utf16(&mut buf).iter().rev() {
            if i == 0 || s[i - 1] != u {
                return false;
            }
            i -= 1;
        }
    }
    true
}

/// `endsWith(char[] s, int len, char[] suffix)`, compared from the end (the
/// usual miss is the last unit) without a `memcmp` call.
#[inline(always)]
pub fn ends_with_units(s: &[u16], len: usize, suffix: &[u16]) -> bool {
    suffix.len() <= len
        && s[len - suffix.len()..len]
            .iter()
            .rev()
            .zip(suffix.iter().rev())
            .all(|(a, b)| a == b)
}

/// The UTF-16 length of `s`, for [`utf16_units`] at compile time.
pub const fn utf16_len(s: &str) -> usize {
    let b = s.as_bytes();
    let (mut i, mut n) = (0, 0);
    while i < b.len() {
        let x = b[i];
        if x < 0x80 {
            i += 1;
        } else if x < 0xE0 {
            i += 2;
        } else if x < 0xF0 {
            i += 3;
        } else {
            i += 4;
            n += 1;
        }
        n += 1;
    }
    n
}

/// `s` as UTF-16 units at compile time (`N` is [`utf16_len`]): the stemmers'
/// suffix literals, which Java keeps as `char` data.
pub const fn utf16_units<const N: usize>(s: &str) -> [u16; N] {
    let b = s.as_bytes();
    let mut out = [0u16; N];
    let (mut i, mut n) = (0, 0);
    while i < b.len() {
        let x = b[i] as u32;
        let (cp, width) = if x < 0x80 {
            (x, 1)
        } else if x < 0xE0 {
            (((x & 0x1F) << 6) | cont(b, i, 1), 2)
        } else if x < 0xF0 {
            (((x & 0x0F) << 12) | (cont(b, i, 1) << 6) | cont(b, i, 2), 3)
        } else {
            let hi = ((x & 0x07) << 18) | (cont(b, i, 1) << 12);
            (hi | (cont(b, i, 2) << 6) | cont(b, i, 3), 4)
        };
        i += width;
        if cp >= 0x10000 {
            out[n] = (0xD800 + ((cp - 0x10000) >> 10)) as u16;
            out[n + 1] = (0xDC00 + ((cp - 0x10000) & 0x3FF)) as u16;
            n += 2;
        } else {
            out[n] = cp as u16;
            n += 1;
        }
    }
    out
}

/// The payload bits of `b[i + k]`, a UTF-8 continuation byte.
const fn cont(b: &[u8], i: usize, k: usize) -> u32 {
    (b[i + k] & 0x3F) as u32
}

/// A string literal as a `&'static [u16; N]`, converted at compile time.
macro_rules! utf16 {
    ($s:literal) => {{
        const A: [u16; $crate::util::stemmer_util::utf16_len($s)] =
            $crate::util::stemmer_util::utf16_units($s);
        &A
    }};
}
pub(crate) use utf16;

/// `endsWith(s, len, "literal")` over the literal's compile-time units.
macro_rules! ends {
    ($s:expr, $len:expr, $suffix:literal) => {
        $crate::util::stemmer_util::ends_with_units(
            $s,
            $len,
            $crate::util::stemmer_util::utf16!($suffix),
        )
    };
}
pub(crate) use ends;

/// `delete(char[] s, int pos, int len)`: removes `s[pos]`, returns the new
/// length.
#[inline]
pub fn delete(s: &mut [u16], pos: usize, len: usize) -> usize {
    debug_assert!(pos < len);
    if pos < len - 1 {
        s.copy_within(pos + 1..len, pos);
    }
    len - 1
}

/// `deleteN(char[] s, int pos, int len, int nChars)`.
#[inline]
pub fn delete_n(s: &mut [u16], pos: usize, len: usize, n: usize) -> usize {
    debug_assert!(pos + n <= len);
    if pos + n < len {
        s.copy_within(pos + n..len, pos);
    }
    len - n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn prefixes_suffixes_and_deletes() {
        let mut s = u("abcdef");
        assert!(starts_with(&s, 6, "abc"));
        assert!(!starts_with(&s, 2, "abc"));
        assert!(!starts_with(&s, 6, "abd"));
        assert!(ends_with(&s, 6, "ef"));
        assert!(ends_with(&s, 4, "cd"));
        assert!(!ends_with(&s, 1, "ab"));
        assert!(ends_with(&u("x\u{1F600}"), 3, "\u{1F600}"));
        assert!(ends_with(&u("\u{1F600}"), 2, "\u{1F600}"));
        assert!(!ends_with(&u("\u{1F600}"), 1, "\u{1F600}"));
        assert!(!ends_with(&u("a\u{1F600}"), 3, "b\u{1F600}"));
        assert!(!ends_with(&u("ab"), 2, "\u{1F600}"));
        assert!(ends_with(&u("αβγ"), 3, "βγ"));
        assert!(!ends_with(&u("αβγ"), 3, "αγ"));
        assert!(ends_with(&u("कमल"), 3, "मल"));
        assert!(!ends_with(&u("γ"), 1, "αγ"));
        assert!(!ends_with(&u("ab"), 2, "\u{0101}b"));
        assert!(!ends_with(&u("कb"), 2, "ab"));
        assert_eq!(utf16_len("aé\u{915}\u{1F600}"), 5);
        assert_eq!(
            utf16_units::<5>("aé\u{915}\u{1F600}")[..],
            u("aé\u{915}\u{1F600}")[..]
        );
        assert!(ends!(&u("xaé\u{1F600}"), 5, "aé\u{1F600}"));
        assert!(!ends!(&u("xaé"), 3, "bé"));
        assert!(!ends!(&u("é"), 1, "aé"));
        assert!(ends_with_units(&s, 6, &u("def")));
        assert!(!ends_with_units(&s, 2, &u("def")));
        assert_eq!(delete(&mut s, 1, 6), 5);
        assert_eq!(&s[..5], &u("acdef")[..]);
        assert_eq!(delete(&mut s, 4, 5), 4);
        assert_eq!(delete_n(&mut s, 0, 4, 2), 2);
        assert_eq!(&s[..2], &u("de")[..]);
        assert_eq!(delete_n(&mut s, 1, 2, 1), 1);
    }
}
