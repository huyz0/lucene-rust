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

use crate::simple_to_lowercase;

/// `CharArraySet`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CharArraySet {
    words: HashSet<String>,
    ignore_case: bool,
}

/// Lowercases `s` code point by code point with Java's simple mapping,
/// borrowing when nothing changes.
fn lower(s: &str) -> std::borrow::Cow<'_, str> {
    if s.bytes().all(|b| !b.is_ascii_uppercase() && b < 0x80) {
        return std::borrow::Cow::Borrowed(s);
    }
    std::borrow::Cow::Owned(s.chars().map(simple_to_lowercase).collect())
}

impl CharArraySet {
    /// `new CharArraySet(startSize, ignoreCase)`.
    pub fn new(ignore_case: bool) -> Self {
        CharArraySet {
            words: HashSet::new(),
            ignore_case,
        }
    }

    /// `new CharArraySet(startSize, ignoreCase)` with a capacity hint.
    pub fn with_capacity(start_size: usize, ignore_case: bool) -> Self {
        CharArraySet {
            words: HashSet::with_capacity(start_size),
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
            words: words.clone(),
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
