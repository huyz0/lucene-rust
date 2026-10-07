//! The `java.lang.String`/`Character` operations the encoders are written
//! in, over UTF-16 units as Java's `char`s: every encoder indexes, slices and
//! compares `char`s, so the port keeps them as `u16` and only converts at the
//! filter boundary ([`units`], [`string`]).

use lucene_analysis::java_character;
use lucene_analysis::lang::{java_string_to_lower_case, java_string_to_upper_case};

/// A Rust string as Java `char`s.
pub fn units(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// Java `char`s as a Rust string; an unpaired surrogate becomes U+FFFD (what
/// a term's `BytesRef` makes of it).
pub fn string(u: &[u16]) -> String {
    String::from_utf16_lossy(u)
}

/// `Character.isLetter(char)`: a surrogate unit is not a letter.
pub fn is_letter(c: u16) -> bool {
    java_character::is_letter(u32::from(c))
}

/// `Character.isWhitespace(char)`.
pub fn is_whitespace(c: u16) -> bool {
    java_character::is_whitespace(u32::from(c))
}

/// `Character.toLowerCase(char)`: the simple mapping of a BMP unit (a
/// surrogate maps to itself).
pub fn to_lower_char(c: u16) -> u16 {
    // A BMP character's simple lowercase is a BMP character.
    java_character::to_lower_case(u32::from(c)) as u16
}

/// `Character.toUpperCase(char)`.
pub fn to_upper_char(c: u16) -> u16 {
    java_character::to_upper_case(u32::from(c)) as u16
}

/// `String.toUpperCase(Locale.ENGLISH)` (`ß` -> `SS`).
pub fn to_upper(s: &[u16]) -> Vec<u16> {
    java_string_to_upper_case(s)
}

/// `String.toLowerCase(Locale.ENGLISH)`.
pub fn to_lower(s: &[u16]) -> Vec<u16> {
    java_string_to_lower_case(s)
}

/// `String.trim()`: strips every unit `<= ' '` from both ends.
pub fn trim(s: &[u16]) -> &[u16] {
    let start = s.iter().position(|&c| c > 0x20).unwrap_or(s.len());
    let end = s.iter().rposition(|&c| c > 0x20).map_or(start, |e| e + 1);
    &s[start..end]
}

/// `str.startsWith(prefix)` over units.
pub fn starts_with(s: &[u16], prefix: &str) -> bool {
    let mut it = s.iter();
    prefix.encode_utf16().all(|p| it.next() == Some(&p))
}

/// `String.replace(CharSequence, CharSequence)`: every non-overlapping
/// occurrence, left to right. `from` is never empty here.
pub fn replace(s: &[u16], from: &[u16], to: &[u16]) -> Vec<u16> {
    debug_assert!(!from.is_empty());
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// `java.util.regex`'s `\s` (no `UNICODE_CHARACTER_CLASS`): `[ \t\n\x0B\f\r]`.
pub fn is_regex_space(c: u16) -> bool {
    matches!(c, 0x20 | 0x09 | 0x0A | 0x0B | 0x0C | 0x0D)
}

/// `str.split("\\s+")`: Java's `String.split` -- a leading empty string
/// kept when the input starts with a separator (unless the match is at 0
/// and zero-length, which `\s+` never is), trailing empty strings removed,
/// and a string with no separator returned whole (so `""` gives `[""]`).
pub fn split_whitespace(s: &[u16]) -> Vec<&[u16]> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < s.len() {
        if is_regex_space(s[i]) {
            let mut j = i;
            while j < s.len() && is_regex_space(s[j]) {
                j += 1;
            }
            parts.push(&s[start..i]);
            start = j;
            i = j;
        } else {
            i += 1;
        }
    }
    if parts.is_empty() {
        return vec![s];
    }
    parts.push(&s[start..]);
    while parts.len() > 1 && parts.last().is_some_and(|p| p.is_empty()) {
        parts.pop();
    }
    if parts.len() == 1 && parts[0].is_empty() {
        // Every part was empty: Java's split returns an empty array then.
        parts.clear();
    }
    parts
}

/// `String.hashCode()` over units.
pub fn string_hash(s: &[u16]) -> i32 {
    s.iter()
        .fold(0i32, |h, &c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// The order a `java.util.HashSet<String>`/`HashMap<String, _>` built by
/// inserting `keys` in order iterates them: by bucket
/// (`(h ^ (h >>> 16)) & (n - 1)` for a table of `n`), then insertion order
/// within a bucket. `initial_capacity` is the table size the collection
/// starts with (16 for `new HashMap<>()`; `new HashSet<>(c)` sizes for
/// `c.size() / .75f + 1`); the table doubles whenever the size exceeds
/// three quarters of it. Equal keys are inserted once.
pub fn java_hash_order<'a>(keys: &[&'a [u16]], initial_capacity: usize) -> Vec<&'a [u16]> {
    let mut distinct: Vec<&'a [u16]> = Vec::new();
    for k in keys {
        if !distinct.contains(k) {
            distinct.push(k);
        }
    }
    let mut n = initial_capacity.next_power_of_two().max(1);
    while distinct.len() * 4 > n * 3 {
        n *= 2;
    }
    let bucket = |k: &[u16]| {
        let h = string_hash(k) as u32;
        ((h ^ (h >> 16)) as usize) & (n - 1)
    };
    let mut order: Vec<(usize, usize)> = distinct
        .iter()
        .enumerate()
        .map(|(i, k)| (bucket(k), i))
        .collect();
    order.sort_unstable();
    order.into_iter().map(|(_, i)| distinct[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Vec<u16> {
        units(s)
    }

    #[test]
    fn trim_and_split_as_java() {
        assert_eq!(trim(&u("  a b\t\u{1}")), u("a b").as_slice());
        assert!(trim(&u(" \t ")).is_empty());
        let split =
            |s: &str| -> Vec<String> { split_whitespace(&u(s)).into_iter().map(string).collect() };
        assert_eq!(split(""), [""]);
        assert_eq!(split("ab"), ["ab"]);
        assert_eq!(split("a  b\tc"), ["a", "b", "c"]);
        assert_eq!(split(" a b "), ["", "a", "b"]);
        assert!(split("   ").is_empty());
    }

    #[test]
    fn replace_is_left_to_right() {
        assert_eq!(replace(&u("aaa"), &u("aa"), &u("b")), u("ba"));
        assert_eq!(replace(&u("xyz"), &u("q"), &u("")), u("xyz"));
        assert!(starts_with(&u("abc"), "ab"));
        assert!(!starts_with(&u("a"), "ab"));
    }

    #[test]
    fn hash_order_follows_java_buckets() {
        // Inserted b, a, q into 16 buckets: a and q share bucket 1 (97 & 15,
        // 113 & 15), b is in bucket 2.
        let keys = [u("b"), u("a"), u("q")];
        let refs: Vec<&[u16]> = keys.iter().map(Vec::as_slice).collect();
        let order: Vec<String> = java_hash_order(&refs, 16).into_iter().map(string).collect();
        assert_eq!(order, ["a", "q", "b"]);
        assert_eq!(
            string_hash(&u("de la")),
            "de la"
                .chars()
                .fold(0i32, |h, c| { h.wrapping_mul(31).wrapping_add(c as i32) })
        );
        // Thirteen keys outgrow 16 buckets: in 32, 0, 32, 64 ... share bucket
        // 0 and 16, 48 ... bucket 16.
        let many: Vec<Vec<u16>> = (0..13u16).map(|i| vec![i * 16]).collect();
        let refs: Vec<&[u16]> = many.iter().map(Vec::as_slice).collect();
        let order = java_hash_order(&refs, 16);
        assert_eq!(order[1], [32u16].as_slice());
        assert_eq!(order[7], [16u16].as_slice());
    }

    #[test]
    fn char_helpers() {
        assert!(is_letter(u16::from(b'a')));
        assert!(!is_letter(0xD800));
        assert!(is_whitespace(0x20));
        assert_eq!(to_lower_char(u16::from(b'A')), u16::from(b'a'));
        assert_eq!(to_upper_char(u16::from(b'a')), u16::from(b'A'));
        assert_eq!(to_upper(&u("ß")), u("SS"));
        assert_eq!(to_lower(&u("İ")), u("i\u{307}"));
        assert!(is_regex_space(0x0B));
        assert!(!is_regex_space(0xA0));
    }
}
