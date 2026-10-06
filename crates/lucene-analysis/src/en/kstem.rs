//! `org.apache.lucene.analysis.en.KStemmer` and `KStemFilter`: Krovetz's
//! dictionary-driven stemmer, ported method for method.
//!
//! The dictionary (`dict_ht`) is built once from [`super::kstem_data`] in
//! Java's insertion order with Java's duplicate checks. `word` is Java's
//! `OpenStringBuilder`, kept as a buffer plus a length because KStem
//! depends on its quirk: `setLength` past the current length re-exposes the
//! characters still in the buffer, including from the previous word, and
//! the buffer lives as long as the stemmer.

use std::collections::HashMap;
use std::sync::LazyLock;

use super::kstem_data::{
    COUNTRY_NATIONALITY, DIRECT_CONFLATIONS, EXCEPTION_WORDS, HEAD_WORDS, PROPER_NOUNS,
    SUPPLEMENT_DICT,
};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `KStemmer.MaxWordLen`.
const MAX_WORD_LEN: usize = 50;

/// `KStemmer.DictEntry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DictEntry {
    exception: bool,
    root: Option<&'static str>,
}

/// `KStemmer.dict_ht` (a `CharArrayMap` over UTF-16 units; every key is
/// ASCII, so a `str` key is the same lookup).
static DICT: LazyLock<HashMap<&'static str, DictEntry>> = LazyLock::new(|| {
    let mut d = HashMap::with_capacity(30_000);
    let mut put = |k: &'static str, e: DictEntry, which: u8| {
        let fresh = d.insert(k, e).is_none();
        assert!(fresh, "Warning: Entry [{k}] already in dictionary {which}");
    };
    for &w in EXCEPTION_WORDS {
        put(
            w,
            DictEntry {
                exception: true,
                root: Some(w),
            },
            1,
        );
    }
    for &(w, root) in DIRECT_CONFLATIONS {
        put(
            w,
            DictEntry {
                exception: false,
                root: Some(root),
            },
            2,
        );
    }
    for &(w, root) in COUNTRY_NATIONALITY {
        put(
            w,
            DictEntry {
                exception: false,
                root: Some(root),
            },
            3,
        );
    }
    let default = DictEntry {
        exception: false,
        root: None,
    };
    for &w in HEAD_WORDS {
        put(w, default, 4);
    }
    for &w in SUPPLEMENT_DICT {
        put(w, default, 5);
    }
    for &w in PROPER_NOUNS {
        put(w, default, 6);
    }
    d
});

fn dict_get(chars: &[u16]) -> Option<DictEntry> {
    // Every key is ASCII: a word with any other unit is absent.
    if chars.iter().any(|&c| c >= 0x80) {
        return None;
    }
    let s: String = chars.iter().map(|&c| c as u8 as char).collect();
    DICT.get(s.as_str()).copied()
}

/// `org.apache.lucene.analysis.util.OpenStringBuilder`, as KStem uses it.
#[derive(Debug, Clone)]
struct OpenStringBuilder {
    buf: Vec<u16>,
    len: usize,
}

impl OpenStringBuilder {
    fn new() -> Self {
        OpenStringBuilder {
            buf: vec![0; 32],
            len: 0,
        }
    }

    fn resize(&mut self, len: usize) {
        let mut nb = vec![0u16; (self.buf.len() << 1).max(len)];
        nb[..self.len].copy_from_slice(&self.buf[..self.len]);
        self.buf = nb;
    }

    fn reserve(&mut self, num: usize) {
        if self.len + num > self.buf.len() {
            self.resize(self.len + num);
        }
    }

    fn set_length(&mut self, len: usize) {
        self.len = len;
    }

    fn char_at(&self, i: usize) -> u16 {
        self.buf[i]
    }

    fn set_char_at(&mut self, i: usize, c: u8) {
        self.buf[i] = u16::from(c);
    }

    fn unsafe_write(&mut self, c: u16) {
        if self.len >= self.buf.len() {
            // Java would throw; the caller reserved enough. Grow instead.
            self.resize(self.len + 1);
        }
        self.buf[self.len] = c;
        self.len += 1;
    }

    fn append(&mut self, s: &str) {
        self.reserve(s.len());
        for b in s.bytes() {
            self.unsafe_write(u16::from(b));
        }
    }

    fn as_slice(&self) -> &[u16] {
        &self.buf[..self.len]
    }
}

fn is(c: u16, ch: u8) -> bool {
    c == u16::from(ch)
}

/// `org.apache.lucene.analysis.en.KStemmer`.
#[derive(Debug, Clone)]
pub struct KStemmer {
    word: OpenStringBuilder,
    j: i32,
    k: i32,
    matched_entry: Option<DictEntry>,
    result: Option<&'static str>,
}

impl Default for KStemmer {
    fn default() -> Self {
        Self::new()
    }
}

impl KStemmer {
    /// `new KStemmer()`.
    pub fn new() -> Self {
        KStemmer {
            word: OpenStringBuilder::new(),
            j: 0,
            k: 0,
            matched_entry: None,
            result: None,
        }
    }

    fn ch(&self, i: i32) -> u16 {
        self.word.char_at(i as usize)
    }

    fn penult_char(&self) -> u16 {
        self.ch(self.k - 1)
    }

    fn is_vowel(&self, i: i32) -> bool {
        !self.is_cons(i)
    }

    fn is_cons(&self, i: i32) -> bool {
        let ch = self.ch(i);
        if is(ch, b'a') || is(ch, b'e') || is(ch, b'i') || is(ch, b'o') || is(ch, b'u') {
            return false;
        }
        if !is(ch, b'y') || i == 0 {
            return true;
        }
        !self.is_cons(i - 1)
    }

    fn set_len(&mut self, n: i32) {
        self.word.set_length(n as usize);
    }

    fn write(&mut self, c: u8) {
        self.word.unsafe_write(u16::from(c));
    }

    fn write_unit(&mut self, c: u16) {
        self.word.unsafe_write(c);
    }

    fn set_at(&mut self, i: i32, c: u8) {
        self.word.set_char_at(i as usize, c);
    }

    fn stem_length(&self) -> i32 {
        self.j + 1
    }

    // Java: endsIn(char[])
    fn ends_in_str(&mut self, s: &str) -> bool {
        let s = s.as_bytes();
        if s.len() as i32 > self.k {
            return false;
        }
        let r = self.word.len as i32 - s.len() as i32;
        self.j = self.k;
        for (i, &c) in s.iter().enumerate() {
            if !is(self.ch(r + i as i32), c) {
                return false;
            }
        }
        self.j = r - 1;
        true
    }

    // Java: endsIn(char a, char b[, ...])
    fn ends_in(&mut self, s: &[u8]) -> bool {
        let n = s.len() as i32;
        if n > self.k {
            return false;
        }
        for (i, &c) in s.iter().enumerate() {
            if !is(self.ch(self.k - n + 1 + i as i32), c) {
                return false;
            }
        }
        self.j = self.k - n;
        true
    }

    // Java: wordInDict
    fn word_in_dict(&mut self) -> Option<DictEntry> {
        if self.matched_entry.is_some() {
            return self.matched_entry;
        }
        let e = dict_get(self.word.as_slice());
        if let Some(e) = e {
            if !e.exception {
                self.matched_entry = Some(e);
            }
        }
        e
    }

    // Java: lookup
    fn lookup(&mut self) -> bool {
        self.matched_entry = dict_get(self.word.as_slice());
        self.matched_entry.is_some()
    }

    // Java: setSuffix / setSuff
    fn set_suffix(&mut self, s: &str) {
        self.set_len(self.j + 1);
        for b in s.bytes() {
            self.write(b);
        }
        self.k = self.j + s.len() as i32;
    }

    // Java: plural
    fn plural(&mut self) {
        if !is(self.ch(self.k), b's') {
            return;
        }
        if self.ends_in(b"ies") {
            self.set_len(self.j + 3);
            self.k -= 1;
            if self.lookup() {
                return;
            }
            self.k += 1;
            self.write(b's');
            self.set_suffix("y");
            self.lookup();
        } else if self.ends_in(b"es") {
            self.set_len(self.j + 2);
            self.k -= 1;
            let try_e = self.j > 0 && !(is(self.ch(self.j), b's') && is(self.ch(self.j - 1), b's'));
            if try_e && self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.k -= 1;
            if self.lookup() {
                return;
            }
            self.write(b'e');
            self.k += 1;
            if !try_e {
                self.lookup();
            }
        } else if self.word.len > 3 && !is(self.penult_char(), b's') && !self.ends_in(b"ous") {
            self.set_len(self.k);
            self.k -= 1;
            self.lookup();
        }
    }

    // Java: pastTense
    fn past_tense(&mut self) {
        if self.word.len <= 4 {
            return;
        }
        if self.ends_in(b"ied") {
            self.set_len(self.j + 3);
            self.k -= 1;
            if self.lookup() {
                return;
            }
            self.k += 1;
            self.write(b'd');
            self.set_suffix("y");
            self.lookup();
            return;
        }
        if self.ends_in(b"ed") && self.vowel_in_stem() {
            self.set_len(self.j + 2);
            self.k = self.j + 1;
            if let Some(entry) = self.word_in_dict() {
                if !entry.exception {
                    return;
                }
            }
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            if self.double_c(self.k) {
                self.set_len(self.k);
                self.k -= 1;
                if self.lookup() {
                    return;
                }
                let c = self.ch(self.k);
                self.write_unit(c);
                self.k += 1;
                self.lookup();
                return;
            }
            if is(self.ch(0), b'u') && is(self.ch(1), b'n') {
                self.write(b'e');
                self.write(b'd');
                self.k += 2;
                return;
            }
            self.set_len(self.j + 1);
            self.write(b'e');
            self.k = self.j + 1;
        }
    }

    // Java: doubleC
    fn double_c(&self, i: i32) -> bool {
        if i < 1 {
            return false;
        }
        if self.ch(i) != self.ch(i - 1) {
            return false;
        }
        self.is_cons(i)
    }

    // Java: vowelInStem
    fn vowel_in_stem(&self) -> bool {
        (0..self.stem_length()).any(|i| self.is_vowel(i))
    }

    // Java: aspect
    fn aspect(&mut self) {
        if self.word.len <= 5 {
            return;
        }
        if self.ends_in(b"ing") && self.vowel_in_stem() {
            self.set_at(self.j + 1, b'e');
            self.set_len(self.j + 2);
            self.k = self.j + 1;
            if let Some(entry) = self.word_in_dict() {
                if !entry.exception {
                    return;
                }
            }
            self.set_len(self.k);
            self.k -= 1;
            if self.lookup() {
                return;
            }
            if self.double_c(self.k) {
                self.k -= 1;
                self.set_len(self.k + 1);
                if self.lookup() {
                    return;
                }
                let c = self.ch(self.k);
                self.write_unit(c);
                self.k += 1;
                self.lookup();
                return;
            }
            if self.j > 0 && self.is_cons(self.j) && self.is_cons(self.j - 1) {
                self.k = self.j;
                self.set_len(self.k + 1);
                return;
            }
            self.set_len(self.j + 1);
            self.write(b'e');
            self.k = self.j + 1;
        }
    }

    // Java: ityEndings
    fn ity_endings(&mut self) {
        let old_k = self.k;
        if self.ends_in(b"ity") {
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.write(b'e');
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_at(self.j + 1, b'i');
            self.word.append("ty");
            self.k = old_k;
            let j = self.j;
            if j > 0 && is(self.ch(j - 1), b'i') && is(self.ch(j), b'l') {
                self.set_len(j - 1);
                self.word.append("le");
                self.k = j;
                self.lookup();
                return;
            }
            if j > 0 && is(self.ch(j - 1), b'i') && is(self.ch(j), b'v') {
                self.set_len(j + 1);
                self.write(b'e');
                self.k = j + 1;
                self.lookup();
                return;
            }
            if j > 0 && is(self.ch(j - 1), b'a') && is(self.ch(j), b'l') {
                self.set_len(j + 1);
                self.k = j;
                self.lookup();
                return;
            }
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.k = j;
        }
    }

    // Java: nceEndings
    fn nce_endings(&mut self) {
        let old_k = self.k;
        if self.ends_in(b"nce") {
            let word_char = self.ch(self.j);
            if !(is(word_char, b'e') || is(word_char, b'a')) {
                return;
            }
            self.set_len(self.j);
            self.write(b'e');
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.set_len(self.j);
            self.k = self.j - 1;
            if self.lookup() {
                return;
            }
            self.write_unit(word_char);
            self.word.append("nce");
            self.k = old_k;
        }
    }

    // Java: nessEndings
    fn ness_endings(&mut self) {
        if self.ends_in(b"ness") {
            self.set_len(self.j + 1);
            self.k = self.j;
            if is(self.ch(self.j), b'i') {
                self.set_at(self.j, b'y');
            }
            self.lookup();
        }
    }

    // Java: ismEndings
    fn ism_endings(&mut self) {
        if self.ends_in(b"ism") {
            self.set_len(self.j + 1);
            self.k = self.j;
            self.lookup();
        }
    }

    // Java: mentEndings
    fn ment_endings(&mut self) {
        let old_k = self.k;
        if self.ends_in(b"ment") {
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.word.append("ment");
            self.k = old_k;
        }
    }

    // Java: izeEndings
    fn ize_endings(&mut self) {
        let old_k = self.k;
        if self.ends_in(b"ize") {
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.write(b'i');
            if self.double_c(self.j) {
                self.set_len(self.j);
                self.k = self.j - 1;
                if self.lookup() {
                    return;
                }
                let c = self.ch(self.j - 1);
                self.write_unit(c);
            }
            self.set_len(self.j + 1);
            self.write(b'e');
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.word.append("ize");
            self.k = old_k;
        }
    }

    // Java: ncyEndings
    fn ncy_endings(&mut self) {
        if self.ends_in(b"ncy") {
            if !(is(self.ch(self.j), b'e') || is(self.ch(self.j), b'a')) {
                return;
            }
            self.set_at(self.j + 2, b't');
            self.set_len(self.j + 3);
            self.k = self.j + 2;
            if self.lookup() {
                return;
            }
            self.set_at(self.j + 2, b'c');
            self.write(b'e');
            self.k = self.j + 3;
            self.lookup();
        }
    }

    // Java: bleEndings
    fn ble_endings(&mut self) {
        let old_k = self.k;
        if self.ends_in(b"ble") {
            if !(is(self.ch(self.j), b'a') || is(self.ch(self.j), b'i')) {
                return;
            }
            let word_char = self.ch(self.j);
            self.set_len(self.j);
            self.k = self.j - 1;
            if self.lookup() {
                return;
            }
            if self.double_c(self.k) {
                self.set_len(self.k);
                self.k -= 1;
                if self.lookup() {
                    return;
                }
                self.k += 1;
                let c = self.ch(self.k - 1);
                self.write_unit(c);
            }
            self.set_len(self.j);
            self.write(b'e');
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.set_len(self.j);
            self.word.append("ate");
            self.k = self.j + 2;
            if self.lookup() {
                return;
            }
            self.set_len(self.j);
            self.write_unit(word_char);
            self.word.append("ble");
            self.k = old_k;
        }
    }

    // Java: icEndings
    fn ic_endings(&mut self) {
        if self.ends_in(b"ic") {
            self.set_len(self.j + 3);
            self.word.append("al");
            self.k = self.j + 4;
            if self.lookup() {
                return;
            }
            self.set_at(self.j + 1, b'y');
            self.set_len(self.j + 2);
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_at(self.j + 1, b'e');
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.word.append("ic");
            self.k = self.j + 2;
        }
    }

    // Java: ionEndings
    fn ion_endings(&mut self) {
        let old_k = self.k;
        if !self.ends_in(b"ion") {
            return;
        }
        if self.ends_in_str("ization") {
            self.set_len(self.j + 3);
            self.write(b'e');
            self.k = self.j + 3;
            self.lookup();
            return;
        }
        if self.ends_in_str("ition") {
            self.set_len(self.j + 1);
            self.write(b'e');
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.word.append("ition");
            self.k = old_k;
        } else if self.ends_in_str("ation") {
            self.set_len(self.j + 3);
            self.write(b'e');
            self.k = self.j + 3;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.write(b'e');
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.word.append("ation");
            self.k = old_k;
        }
        if self.ends_in_str("ication") {
            self.set_len(self.j + 1);
            self.write(b'y');
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.word.append("ication");
            self.k = old_k;
        }
        // Java: `if (true)` -- "we checked for this earlier... just need to set j".
        self.j = self.k - 3;
        self.set_len(self.j + 1);
        self.write(b'e');
        self.k = self.j + 1;
        if self.lookup() {
            return;
        }
        self.set_len(self.j + 1);
        self.k = self.j;
        if self.lookup() {
            return;
        }
        self.set_len(self.j + 1);
        self.word.append("ion");
        self.k = old_k;
    }

    // Java: erAndOrEndings
    fn er_and_or_endings(&mut self) {
        let old_k = self.k;
        if !is(self.ch(self.k), b'r') {
            return;
        }
        if self.ends_in(b"izer") {
            self.set_len(self.j + 4);
            self.k = self.j + 3;
            self.lookup();
            return;
        }
        if self.ends_in(b"er") || self.ends_in(b"or") {
            let word_char = self.ch(self.j + 1);
            if self.double_c(self.j) {
                self.set_len(self.j);
                self.k = self.j - 1;
                if self.lookup() {
                    return;
                }
                let c = self.ch(self.j - 1);
                self.write_unit(c);
            }
            if is(self.ch(self.j), b'i') {
                self.set_at(self.j, b'y');
                self.set_len(self.j + 1);
                self.k = self.j;
                if self.lookup() {
                    return;
                }
                self.set_at(self.j, b'i');
                self.write(b'e');
            }
            if is(self.ch(self.j), b'e') {
                self.set_len(self.j);
                self.k = self.j - 1;
                if self.lookup() {
                    return;
                }
                self.write(b'e');
            }
            self.set_len(self.j + 2);
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.write(b'e');
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.write_unit(word_char);
            self.write(b'r');
            self.k = old_k;
        }
    }

    // Java: lyEndings
    fn ly_endings(&mut self) {
        let old_k = self.k;
        if self.ends_in(b"ly") {
            self.set_at(self.j + 2, b'e');
            if self.lookup() {
                return;
            }
            self.set_at(self.j + 2, b'y');
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            let j = self.j;
            if j > 0 && is(self.ch(j - 1), b'a') && is(self.ch(j), b'l') {
                return;
            }
            self.word.append("ly");
            self.k = old_k;
            if j > 0 && is(self.ch(j - 1), b'a') && is(self.ch(j), b'b') {
                self.set_at(j + 2, b'e');
                self.k = j + 2;
                return;
            }
            if is(self.ch(j), b'i') {
                self.set_len(j);
                self.write(b'y');
                self.k = j;
                if self.lookup() {
                    return;
                }
                self.set_len(j);
                self.word.append("ily");
                self.k = old_k;
            }
            self.set_len(j + 1);
            self.k = j;
        }
    }

    // Java: alEndings
    fn al_endings(&mut self) {
        let old_k = self.k;
        if self.word.len < 4 {
            return;
        }
        if self.ends_in(b"al") {
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            if self.double_c(self.j) {
                self.set_len(self.j);
                self.k = self.j - 1;
                if self.lookup() {
                    return;
                }
                let c = self.ch(self.j - 1);
                self.write_unit(c);
            }
            self.set_len(self.j + 1);
            self.write(b'e');
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.word.append("um");
            self.k = self.j + 2;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.word.append("al");
            self.k = old_k;
            let j = self.j;
            if j > 0 && is(self.ch(j - 1), b'i') && is(self.ch(j), b'c') {
                self.set_len(j - 1);
                self.k = j - 2;
                if self.lookup() {
                    return;
                }
                self.set_len(j - 1);
                self.write(b'y');
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                self.set_len(j - 1);
                self.word.append("ic");
                self.k = j;
                self.lookup();
                return;
            }
            if is(self.ch(j), b'i') {
                self.set_len(j);
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                self.word.append("ial");
                self.k = old_k;
                self.lookup();
            }
        }
    }

    // Java: iveEndings
    fn ive_endings(&mut self) {
        let old_k = self.k;
        if self.ends_in(b"ive") {
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.write(b'e');
            self.k = self.j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.word.append("ive");
            let j = self.j;
            if j > 0 && is(self.ch(j - 1), b'a') && is(self.ch(j), b't') {
                self.set_at(j - 1, b'e');
                self.set_len(j);
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                self.set_len(j - 1);
                if self.lookup() {
                    return;
                }
                self.word.append("ative");
                self.k = old_k;
            }
            self.set_at(j + 2, b'o');
            self.set_at(j + 3, b'n');
            if self.lookup() {
                return;
            }
            self.set_at(j + 2, b'v');
            self.set_at(j + 3, b'e');
            self.k = old_k;
        }
    }

    /// `stem(char[], int)`: whether the term changed; the stem is then
    /// [`Self::as_utf16`].
    pub fn stem(&mut self, term: &[u16]) -> bool {
        self.result = None;
        let len = term.len();
        self.k = len as i32 - 1;
        if self.k <= 1 || self.k >= MAX_WORD_LEN as i32 - 1 {
            return false;
        }
        if let Some(entry) = dict_get(term) {
            if let Some(root) = entry.root {
                self.result = Some(root);
                return true;
            }
            return false;
        }
        self.word.set_length(0);
        self.word.reserve(len + 10);
        for &ch in term {
            // isAlpha: terms must be lowercased already.
            if !(u16::from(b'a')..=u16::from(b'z')).contains(&ch) {
                return false;
            }
            self.word.unsafe_write(ch);
        }
        self.matched_entry = None;
        let steps: [fn(&mut Self); 16] = [
            Self::plural,
            Self::past_tense,
            Self::aspect,
            Self::ity_endings,
            Self::ness_endings,
            Self::ion_endings,
            Self::er_and_or_endings,
            Self::ly_endings,
            Self::al_endings,
            Self::ive_endings,
            Self::ize_endings,
            Self::ment_endings,
            Self::ble_endings,
            Self::ism_endings,
            Self::ic_endings,
            Self::ncy_endings,
        ];
        let mut done = false;
        for (i, step) in steps.iter().enumerate() {
            if i == 9 {
                // Java calls wordInDict() before iveEndings (result unused,
                // but it may cache the match).
                self.word_in_dict();
            }
            step(self);
            if self.matched_entry.is_some() {
                done = true;
                break;
            }
        }
        if !done {
            self.nce_endings();
        }
        if let Some(entry) = self.matched_entry {
            self.result = entry.root;
        }
        true
    }

    /// `asCharSequence()`: the stem of the last [`Self::stem`] that returned
    /// `true`.
    pub fn as_utf16(&self) -> Vec<u16> {
        match self.result {
            Some(r) => r.encode_utf16().collect(),
            None => self.word.as_slice().to_vec(),
        }
    }
}

/// `org.apache.lucene.analysis.en.KStemFilter`: stems every non-keyword term
/// (which must be lowercase already) with [`KStemmer`].
pub struct KStemFilter<I> {
    input: I,
    stemmer: KStemmer,
    buf: Vec<u16>,
}

impl<I: TokenStream> KStemFilter<I> {
    /// `new KStemFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        KStemFilter {
            input,
            stemmer: KStemmer::new(),
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for KStemFilter<I> {
    crate::filter_input!();
    // Java: KStemFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if !a.is_keyword() {
            self.buf.clear();
            self.buf.extend(a.term().encode_utf16());
            if self.stemmer.stem(&self.buf) {
                let stem = self.stemmer.as_utf16();
                a.set_term_utf16(&stem);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::Canned;

    fn stem(words: &[&str]) -> Vec<String> {
        let mut s = KStemmer::new();
        words
            .iter()
            .map(|w| {
                let t: Vec<u16> = w.encode_utf16().collect();
                if s.stem(&t) {
                    String::from_utf16(&s.as_utf16()).unwrap()
                } else {
                    w.to_string()
                }
            })
            .collect()
    }

    #[test]
    fn dictionary_is_built_like_javas() {
        assert!(DICT.len() > 27_000);
        assert_eq!(
            dict_get(&"aide".encode_utf16().collect::<Vec<_>>()).map(|e| e.exception),
            Some(true)
        );
        assert_eq!(dict_get(&[0xE9]), None);
    }

    #[test]
    fn keyword_and_short_terms_pass_through() {
        assert_eq!(
            stem(&["is", "Dogs", "a😀s", &"x".repeat(60), "going", "aide"]),
            vec!["is", "Dogs", "a😀s", &"x".repeat(60), "go", "aide"]
        );
        let mut c = Canned::parse("dogs:0:4:1:1 cats:5:9:1:1");
        c.set_keywords(&[true, false]);
        let mut out = Vec::new();
        crate::token_stream::consume(&mut KStemFilter::new(c), |a| out.push(a.term().to_string()))
            .unwrap();
        assert_eq!(out, vec!["dogs", "cat"]);
    }

    #[test]
    fn internals() {
        let mut s = KStemmer::default();
        assert!(!s.double_c(0));
        // wordInDict returns the cached match.
        s.matched_entry = Some(DictEntry {
            exception: false,
            root: Some("x"),
        });
        assert_eq!(s.word_in_dict().and_then(|e| e.root), Some("x"));
        let mut b = OpenStringBuilder::new();
        for _ in 0..40 {
            b.unsafe_write(u16::from(b'a'));
        }
        assert_eq!(b.len, 40);
    }

    #[test]
    fn open_string_builder_keeps_stale_chars() {
        let mut b = OpenStringBuilder::new();
        b.append("abcdef");
        b.set_length(2);
        b.set_length(4);
        assert_eq!(String::from_utf16(b.as_slice()).unwrap(), "abcd");
        b.set_length(31);
        b.append("xyz");
        assert_eq!(b.len, 34);
        assert_eq!(b.char_at(33), u16::from(b'z'));
    }
}
