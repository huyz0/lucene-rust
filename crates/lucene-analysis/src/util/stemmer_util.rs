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
#[inline]
pub fn ends_with(s: &[u16], len: usize, suffix: &str) -> bool {
    let n = suffix.encode_utf16().count();
    if n > len {
        return false;
    }
    s[len - n..len].iter().copied().eq(suffix.encode_utf16())
}

/// `endsWith(char[] s, int len, char[] suffix)`.
#[inline]
pub fn ends_with_units(s: &[u16], len: usize, suffix: &[u16]) -> bool {
    suffix.len() <= len && &s[len - suffix.len()..len] == suffix
}

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
