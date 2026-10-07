//! `org.apache.lucene.analysis.CharArraySet`: the stopword/word set every
//! analyzer uses, with its `ignoreCase` mode.
//!
//! Java's class is a set of `char[]` keys over `CharArrayMap`, which exists
//! so a filter can probe the set with a slice of its term buffer without
//! allocating a `String`. A Rust `&str` already is that borrowed slice, so the
//! set is a `HashSet<String>` probed with `&str` (`CharArrayMap` itself is
//! recorded `not-needed` for that reason).
//!
//! `ignoreCase` is Java's exactly: `put` lowercases the stored key with
//! `CharacterUtils.toLowerCase` and lookups compare
//! `Character.toLowerCase(codePoint)` -- the *simple*, 1:1 per-code-point
//! mapping, which is [`crate::simple_to_lowercase`], not Rust's full
//! `str::to_lowercase`.
//!
//! `unmodifiableSet` has no counterpart to write: a set shared through an
//! `Arc` (as [`crate::StopwordAnalyzerBase`] and [`crate::StopFilter`] hold
//! it) is already immutable, and `copy` is `Clone`.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};

use crate::simple_to_lowercase;

/// The set's hash: a multiply-rotate over 8-byte words (FxHash's mix).
/// Every probe is a token of the text being analyzed, so the hash is on the
/// hot path; SipHash's flooding resistance buys nothing for a set whose
/// keys are fixed when it is built.
#[derive(Default)]
pub(crate) struct WordHasher(u64);

impl WordHasher {
    #[inline]
    fn mix(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for WordHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let mut w = [0u8; 8];
            w.copy_from_slice(c);
            self.mix(u64::from_le_bytes(w));
        }
        let rest = chunks.remainder();
        let mut w = [0u8; 8];
        w[..rest.len()].copy_from_slice(rest);
        self.mix(u64::from_le_bytes(w) ^ ((rest.len() as u64) << 56));
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.mix(u64::from(i));
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// [`WordHasher`] as a `HashMap`/`HashSet` hasher.
pub(crate) type WordHash = BuildHasherDefault<WordHasher>;

type Words = HashSet<String, WordHash>;

/// `CharArraySet`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CharArraySet {
    words: Words,
    ignore_case: bool,
}

/// Lowercases `s` code point by code point with Java's simple mapping,
/// borrowing when nothing changes (most non-Latin text: a probe of such a
/// word allocates nothing).
fn lower(s: &str) -> std::borrow::Cow<'_, str> {
    match crate::first_to_lowercase(s.as_bytes()) {
        None => std::borrow::Cow::Borrowed(s),
        Some(i) => {
            let mut out = String::with_capacity(s.len());
            out.push_str(&s[..i]);
            out.extend(s[i..].chars().map(simple_to_lowercase));
            std::borrow::Cow::Owned(out)
        }
    }
}

impl CharArraySet {
    /// `new CharArraySet(startSize, ignoreCase)`.
    pub fn new(ignore_case: bool) -> Self {
        CharArraySet {
            words: Words::default(),
            ignore_case,
        }
    }

    /// `new CharArraySet(startSize, ignoreCase)` with a capacity hint.
    pub fn with_capacity(start_size: usize, ignore_case: bool) -> Self {
        CharArraySet {
            words: Words::with_capacity_and_hasher(start_size, Default::default()),
            ignore_case,
        }
    }

    /// `new CharArraySet(Collection, ignoreCase)`.
    pub fn from_words<S: AsRef<str>>(
        words: impl IntoIterator<Item = S>,
        ignore_case: bool,
    ) -> Self {
        let mut set = CharArraySet::new(ignore_case);
        for w in words {
            set.add(w.as_ref());
        }
        set
    }

    /// `CharArraySet.EMPTY_SET`.
    pub fn empty() -> Self {
        CharArraySet::new(false)
    }

    /// Whether this set was built with `ignoreCase`.
    pub fn ignore_case(&self) -> bool {
        self.ignore_case
    }

    /// `add(String)`: `true` if the word was not already present.
    pub fn add(&mut self, word: &str) -> bool {
        if self.ignore_case {
            self.words.insert(lower(word).into_owned())
        } else {
            self.words.insert(word.to_string())
        }
    }

    /// `contains(CharSequence)`.
    pub fn contains(&self, word: &str) -> bool {
        // StandardAnalyzer's default set is empty: no hash for that.
        if self.words.is_empty() {
            return false;
        }
        if self.ignore_case {
            self.words.contains(lower(word).as_ref())
        } else {
            self.words.contains(word)
        }
    }

    /// `contains(char[] text, int off, int len)`: a probe with a UTF-16
    /// slice, as Java's filters probe with a slice of their term buffer.
    /// The slice is transcoded into a stack buffer (a `String` only past 128
    /// bytes of UTF-8), so a probe allocates nothing. A slice with an
    /// unpaired surrogate is never in the set: Java compares `char`s, and no
    /// key -- a Rust string -- holds one.
    pub fn contains_utf16(&self, word: &[u16]) -> bool {
        if self.words.is_empty() {
            return false;
        }
        let mut buf = [0u8; 128];
        let mut len = 0;
        for c in char::decode_utf16(word.iter().copied()) {
            let Ok(mut c) = c else {
                return false;
            };
            if self.ignore_case {
                c = simple_to_lowercase(c);
            }
            let Some(slot) = buf.get_mut(len..len + c.len_utf8()) else {
                return String::from_utf16(word).is_ok_and(|w| self.contains(&w));
            };
            len += c.encode_utf8(slot).len();
        }
        std::str::from_utf8(&buf[..len]).is_ok_and(|w| self.words.contains(w))
    }

    /// `size()`.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.words.clear();
    }

    /// The stored words (lowercased when `ignoreCase`), in no order.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.words.iter().map(String::as_str)
    }
}

impl From<&HashSet<String>> for CharArraySet {
    /// A case-sensitive set of the same words.
    fn from(words: &HashSet<String>) -> Self {
        CharArraySet {
            words: words.iter().cloned().collect(),
            ignore_case: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_sensitive_by_default() {
        let set = CharArraySet::from_words(["The", "of"], false);
        assert!(set.contains("The"));
        assert!(!set.contains("the"));
        assert_eq!(set.len(), 2);
        assert!(!set.ignore_case());
    }

    #[test]
    fn ignore_case_uses_javas_simple_lowercase() {
        let mut set = CharArraySet::with_capacity(4, true);
        assert!(set.add("ÉCOLE"));
        assert!(!set.add("école"));
        assert!(set.contains("École"));
        assert!(set.contains("école"));
        // Java's simple mapping: U+0130 lowercases to plain `i`.
        set.add("\u{0130}stanbul");
        assert!(set.contains("istanbul"));
        // No final-sigma context rule.
        set.add("ΟΔΟΣ");
        assert!(set.contains("οδοσ"));
        assert!(!set.contains("οδος"));
        let mut words: Vec<&str> = set.iter().collect();
        words.sort();
        assert_eq!(words, vec!["istanbul", "école", "οδοσ"]);
    }

    #[test]
    fn utf16_probes_match_str_probes() {
        let u = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        let long = "ü".repeat(70);
        let mut set = CharArraySet::from_words(["Haus", "😀x", long.as_str(), "\u{FFFD}"], false);
        assert!(set.contains_utf16(&u("Haus")));
        assert!(!set.contains_utf16(&u("haus")));
        assert!(set.contains_utf16(&u("😀x")));
        // 140 bytes of UTF-8: past the stack buffer.
        assert!(set.contains_utf16(&u(&long)));
        assert!(!set.contains_utf16(&u(&"ü".repeat(71))));
        // An unpaired surrogate is not U+FFFD.
        assert!(!set.contains_utf16(&[0xD83D]));
        assert!(!set.contains_utf16(&u("😀x")[1..]));
        set.clear();
        assert!(!set.contains_utf16(&u("Haus")));

        let set = CharArraySet::from_words(["école", "istanbul", long.as_str()], true);
        assert!(set.contains_utf16(&u("ÉCOLE")));
        assert!(set.contains_utf16(&u("\u{0130}stanbul")));
        assert!(set.contains_utf16(&u(&"Ü".repeat(70))));
        assert!(!set.contains_utf16(&u("ecole")));
    }

    #[test]
    fn empty_clear_and_from_hash_set() {
        let mut set = CharArraySet::empty();
        assert!(set.is_empty());
        set.add("x");
        set.clear();
        assert!(set.is_empty());
        let hs: HashSet<String> = ["a".to_string()].into_iter().collect();
        let set = CharArraySet::from(&hs);
        assert!(set.contains("a"));
        assert!(!set.contains("A"));
    }
}
