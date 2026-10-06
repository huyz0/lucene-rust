//! `org.apache.lucene.analysis.hunspell.Dictionary`: parses a Hunspell
//! `.aff` file and its `.dic` files.
//!
//! Text is held as Java `char`s (`u16`) throughout, so string positions,
//! sorting (`String.compareTo`), hashing and case mapping are Lucene's.
//!
//! Differs:
//! - The prefix and suffix tables are tries over the affix code points (the
//!   key Lucene's `FST<IntsRef>` takes, walked unit by unit as Lucene walks
//!   it), not FSTs; lookups return the same affix ids in the same order.
//! - Entries are sorted in memory (`SortingStrategy.inMemory`); the offline
//!   strategy's temporary files have no counterpart.
//! - `SET` decodes UTF-8, ISO-8859-1 (the default), Lucene's `ISO8859-14`
//!   and the JDK's ISO-8859-2, -7, -13, -15, KOI8-R, windows-1251 and
//!   TIS-620 (`charsets.rs`, generated from the JDK), under every name the
//!   JDK accepts; another charset the JDK knows is
//!   [`HunspellError::Unsupported`]. Files are decoded whole: a byte the
//!   charset cannot map fails the load even past a parse error Java's
//!   8 KB reads would have met first.
//! - The `tolerate*` hooks are fixed at Lucene's defaults (`false`) and
//!   `hashFactor` at `1.0`.

use std::collections::{BTreeMap, HashMap};

use super::affix_condition::{unique_key, AffixCondition, AffixKind, ALWAYS_TRUE_KEY};
use super::charsets;
use super::conv_table::ConvTable;
use super::flags::{FlagEnumerator, FlagLookup, FlagParsing};
use super::word_case::{to_lower, to_upper, WordCase};
use super::word_storage::{WordStorage, WordStorageBuilder};
use super::{HunspellError, FLAG_UNSET, HIDDEN_FLAG};
use crate::java_character as jc;

/// `Dictionary.MAX_PROLOGUE_SCAN_WINDOW`.
const MAX_PROLOGUE_SCAN_WINDOW: usize = 30 * 1024;
/// `Dictionary.FLAG_SEPARATOR`.
const FLAG_SEPARATOR: u16 = 0x1f;
/// `Dictionary.MORPH_SEPARATOR`.
const MORPH_SEPARATOR: u16 = 0x1e;

/// `Dictionary.AFFIX_FLAG`.
pub(crate) const AFFIX_FLAG: usize = 0;
/// `Dictionary.AFFIX_STRIP_ORD`.
pub(crate) const AFFIX_STRIP_ORD: usize = 1;
/// `Dictionary.AFFIX_CONDITION`.
const AFFIX_CONDITION: usize = 2;
/// `Dictionary.AFFIX_APPEND`.
pub(crate) const AFFIX_APPEND: usize = 3;

/// UTF-16 units of an ASCII literal.
pub(crate) fn u(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// A lossy `String` of UTF-16 units, for messages.
pub(crate) fn st(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

/// Java's `\s`: `[ \t\n\x0B\f\r]`.
fn is_regex_space(c: u16) -> bool {
    matches!(c, 0x20 | 0x09 | 0x0A | 0x0B | 0x0C | 0x0D)
}

/// `String.split("\\s+")` (`plus`) or `String.split("\\s")`: a leading empty
/// string kept, trailing empty strings removed.
pub(crate) fn java_split(line: &[u16], plus: bool) -> Vec<&[u16]> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < line.len() {
        if is_regex_space(line[i]) {
            parts.push(&line[start..i]);
            i += 1;
            if plus {
                while i < line.len() && is_regex_space(line[i]) {
                    i += 1;
                }
            }
            start = i;
        } else {
            i += 1;
        }
    }
    parts.push(&line[start..]);
    while parts.len() > 1 && parts.last().is_some_and(|p| p.is_empty()) {
        parts.pop();
    }
    if parts.len() == 1 && parts[0].is_empty() && !line.is_empty() {
        // A string of separators only: Java returns an empty array... except
        // that the leading empty string survives when the input is not empty.
        return vec![];
    }
    parts
}

/// `String.trim()`: units `<= ' '` removed at both ends.
pub(crate) fn java_trim(s: &[u16]) -> &[u16] {
    let start = s.iter().position(|&c| c > 0x20).unwrap_or(s.len());
    let end = s.iter().rposition(|&c| c > 0x20).map_or(start, |e| e + 1);
    &s[start..end]
}

/// `String.strip()`: `Character.isWhitespace` units removed at both ends.
fn java_strip(s: &[u16]) -> &[u16] {
    let ws = |c: &u16| jc::is_whitespace(u32::from(*c));
    let start = s.iter().position(|c| !ws(c)).unwrap_or(s.len());
    let end = s.iter().rposition(|c| !ws(c)).map_or(start, |e| e + 1);
    &s[start..end]
}

/// `String.isBlank()`.
fn java_is_blank(s: &[u16]) -> bool {
    s.iter().all(|&c| jc::is_whitespace(u32::from(c)))
}

/// `String.indexOf(char, from)`.
pub(crate) fn index_of(s: &[u16], c: u16, from: usize) -> Option<usize> {
    s.get(from..)?
        .iter()
        .position(|&x| x == c)
        .map(|p| p + from)
}

/// `String.indexOf(String, from)`.
pub(crate) fn index_of_str(s: &[u16], pat: &[u16], from: usize) -> Option<usize> {
    if pat.is_empty() {
        return if from <= s.len() { Some(from) } else { None };
    }
    if from >= s.len() {
        return None;
    }
    s[from..]
        .windows(pat.len())
        .position(|w| w == pat)
        .map(|p| p + from)
}

/// `Integer.parseInt` of UTF-16 units.
fn parse_int(s: &[u16]) -> Result<i32, HunspellError> {
    st(s)
        .parse::<i32>()
        .map_err(|_| HunspellError::NumberFormat(format!("For input string: \"{}\"", st(s))))
}

/// The dictionary's text decoder (`Dictionary.decoder`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Charset {
    Utf8,
    Iso8859_1,
    Iso8859_14,
    /// A JDK single-byte charset: `charsets::TABLES[i]`.
    Table(usize),
}

/// `ISO8859_14Decoder.TABLE`: `0xA0..=0xFF`.
const ISO8859_14: [u16; 96] = [
    0x00A0, 0x1E02, 0x1E03, 0x00A3, 0x010A, 0x010B, 0x1E0A, 0x00A7, 0x1E80, 0x00A9, 0x1E82, 0x1E0B,
    0x1EF2, 0x00AD, 0x00AE, 0x0178, 0x1E1E, 0x1E1F, 0x0120, 0x0121, 0x1E40, 0x1E41, 0x00B6, 0x1E56,
    0x1E81, 0x1E57, 0x1E83, 0x1E60, 0x1EF3, 0x1E84, 0x1E85, 0x1E61, 0x00C0, 0x00C1, 0x00C2, 0x00C3,
    0x00C4, 0x00C5, 0x00C6, 0x00C7, 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF,
    0x0174, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x1E6A, 0x00D8, 0x00D9, 0x00DA, 0x00DB,
    0x00DC, 0x00DD, 0x0176, 0x00DF, 0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7,
    0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF, 0x0175, 0x00F1, 0x00F2, 0x00F3,
    0x00F4, 0x00F5, 0x00F6, 0x1E6B, 0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x0177, 0x00FF,
];

/// Whether `name` passes `Charset.checkName`: ASCII letters and digits,
/// and `-+:_.` after the first character.
fn is_legal_charset_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().enumerate().all(|(i, c)| {
            c.is_ascii_alphanumeric() || i > 0 && matches!(c, b'-' | b'+' | b':' | b'_' | b'.')
        })
}

impl Charset {
    /// `Dictionary.getDecoder`: Lucene's `ISO8859-14`, then
    /// `CHARSET_ALIASES`, then `Charset.forName` (case-insensitive over the
    /// JDK's names and aliases, `charsets::NAMES`). A legal name the JDK
    /// does not know is Java's `UnsupportedCharsetException`, an illegal one
    /// its `IllegalCharsetNameException`; one the JDK knows but this port
    /// cannot decode is [`HunspellError::Unsupported`].
    fn for_name(name: &[u16]) -> Result<Charset, HunspellError> {
        let n = st(name);
        if n == "ISO8859-14" {
            return Ok(Charset::Iso8859_14);
        }
        let n = match n.as_str() {
            "microsoft-cp1251" => "windows-1251".to_string(),
            "TIS620-2533" => "TIS-620".to_string(),
            _ => n,
        };
        let lower = n.to_ascii_lowercase();
        if let Ok(i) = charsets::NAMES.binary_search_by(|(k, _)| (*k).cmp(lower.as_str())) {
            return Ok(match charsets::NAMES[i].1 {
                0 => Charset::Utf8,
                1 => Charset::Iso8859_1,
                t => Charset::Table(t - 2),
            });
        }
        if !is_legal_charset_name(&n) {
            Err(HunspellError::IllegalCharsetName(n))
        } else if charsets::JDK_NAMES.binary_search(&lower.as_str()).is_ok() {
            Err(HunspellError::Unsupported(format!("charset {n}")))
        } else {
            Err(HunspellError::UnsupportedCharset(n))
        }
    }

    /// `decoder.charset()`: Lucene's `ISO8859_14Decoder` reports ISO-8859-1.
    fn java_charset(self) -> Charset {
        match self {
            Charset::Iso8859_14 => Charset::Iso8859_1,
            other => other,
        }
    }

    /// Decodes `bytes` as Lucene's decoder does: malformed UTF-8 replaced
    /// (`CodingErrorAction.REPLACE`), a byte a single-byte charset cannot
    /// map Java's `UnmappableCharacterException` (the action Lucene leaves
    /// at `REPORT`).
    fn decode(self, bytes: &[u8]) -> Result<Vec<u16>, HunspellError> {
        Ok(match self {
            Charset::Utf8 => String::from_utf8_lossy(bytes).encode_utf16().collect(),
            Charset::Iso8859_1 => bytes.iter().map(|&b| u16::from(b)).collect(),
            Charset::Iso8859_14 => bytes
                .iter()
                .map(|&b| {
                    if b >= 0xA0 {
                        ISO8859_14[usize::from(b - 0xA0)]
                    } else {
                        u16::from(b)
                    }
                })
                .collect(),
            Charset::Table(t) => {
                let table = charsets::TABLES[t];
                let mut out = Vec::with_capacity(bytes.len());
                for &b in bytes {
                    let c = if b >= 0x80 {
                        table[usize::from(b - 0x80)]
                    } else {
                        u16::from(b)
                    };
                    if c == charsets::UNMAPPABLE {
                        return Err(HunspellError::UnmappableCharacter(
                            "Input length = 1".to_string(),
                        ));
                    }
                    out.push(c);
                }
                out
            }
        })
    }
}

/// `LineNumberReader` over decoded text: lines split at `\n`, `\r` or
/// `\r\n`, numbered from 1.
pub(crate) struct LineReader {
    text: Vec<u16>,
    pos: usize,
    /// `getLineNumber()`: lines read so far.
    pub(crate) line_number: usize,
}

impl LineReader {
    fn new(text: Vec<u16>) -> Self {
        LineReader {
            text,
            pos: 0,
            line_number: 0,
        }
    }

    /// `readLine()`.
    pub(crate) fn read_line(&mut self) -> Option<Vec<u16>> {
        if self.pos >= self.text.len() {
            return None;
        }
        let start = self.pos;
        let mut end = start;
        while end < self.text.len() && self.text[end] != 0x0A && self.text[end] != 0x0D {
            end += 1;
        }
        let line = self.text[start..end].to_vec();
        self.pos = end;
        if self.pos < self.text.len() {
            if self.text[self.pos] == 0x0D && self.text.get(self.pos + 1) == Some(&0x0A) {
                self.pos += 2;
            } else {
                self.pos += 1;
            }
        }
        self.line_number += 1;
        Some(line)
    }
}

/// `Dictionary.Breaks`.
#[derive(Debug, Clone)]
pub(crate) struct Breaks {
    pub(crate) starting: Vec<Vec<u16>>,
    pub(crate) ending: Vec<Vec<u16>>,
    pub(crate) middle: Vec<Vec<u16>>,
}

impl Breaks {
    fn default_breaks() -> Breaks {
        Breaks {
            starting: vec![u("-")],
            ending: vec![u("-")],
            middle: vec![u("-")],
        }
    }

    /// `Breaks.isNotEmpty`.
    pub(crate) fn is_not_empty(&self) -> bool {
        !self.middle.is_empty() || !self.starting.is_empty() || !self.ending.is_empty()
    }
}

/// `RepEntry`: a `REP` (or `ph:`) replacement for suggestions.
#[derive(Debug, Clone)]
pub(crate) struct RepEntry {
    pattern: Vec<u16>,
    replacement: Vec<u16>,
    must_start: bool,
    must_end: bool,
}

impl RepEntry {
    /// `new RepEntry(rawPattern, rawReplacement)`.
    pub(crate) fn new(raw_pattern: &[u16], raw_replacement: &[u16]) -> RepEntry {
        let must_start = raw_pattern.first() == Some(&u16::from(b'^'));
        let must_end = raw_pattern.last() == Some(&u16::from(b'$'));
        let start = usize::from(must_start);
        let end = raw_pattern.len() - usize::from(must_end);
        RepEntry {
            pattern: raw_pattern[start.min(end)..end].to_vec(),
            replacement: raw_replacement
                .iter()
                .map(|&c| {
                    if c == u16::from(b'_') {
                        u16::from(b' ')
                    } else {
                        c
                    }
                })
                .collect(),
            must_start,
            must_end,
        }
    }

    /// `RepEntry.isMiddle`.
    pub(crate) fn is_middle(&self) -> bool {
        !self.must_start && !self.must_end
    }

    /// `RepEntry.substitute`.
    pub(crate) fn substitute(&self, word: &[u16]) -> Vec<Vec<u16>> {
        let p = &self.pattern;
        if self.must_start {
            let matches = if self.must_end {
                word == p.as_slice()
            } else {
                word.starts_with(p)
            };
            return if matches {
                vec![[self.replacement.as_slice(), &word[p.len()..]].concat()]
            } else {
                vec![]
            };
        }
        if self.must_end {
            return if word.ends_with(p) {
                vec![[&word[..word.len() - p.len()], self.replacement.as_slice()].concat()]
            } else {
                vec![]
            };
        }
        let mut result = Vec::new();
        let mut pos = index_of_str(word, p, 0);
        while let Some(at) = pos {
            result.push(
                [
                    &word[..at],
                    self.replacement.as_slice(),
                    &word[at + p.len()..],
                ]
                .concat(),
            );
            pos = index_of_str(word, p, at + 1);
        }
        result
    }
}

/// `CompoundRule`: a `COMPOUNDRULE` pattern of flags with `?`/`*`.
#[derive(Debug, Clone)]
pub(crate) struct CompoundRule {
    data: Vec<u16>,
}

impl CompoundRule {
    fn new(rule: &[u16], parsing: FlagParsing) -> Result<CompoundRule, HunspellError> {
        let mut parsed = Vec::new();
        let mut pos = 0;
        while pos < rule.len() {
            let Some(l) = index_of(rule, u16::from(b'('), pos) else {
                parsed.extend(parsing.parse_flags(&rule[pos..])?);
                break;
            };
            parsed.extend(parsing.parse_flags(&rule[pos..l])?);
            let Some(r) = index_of(rule, u16::from(b')'), l + 1) else {
                return Err(HunspellError::IllegalArgument(format!(
                    "Unmatched parentheses: {}",
                    st(rule)
                )));
            };
            parsed.extend(parsing.parse_flags(&rule[l + 1..r])?);
            pos = r + 1;
            if pos < rule.len() && (rule[pos] == u16::from(b'?') || rule[pos] == u16::from(b'*')) {
                parsed.push(rule[pos]);
                pos += 1;
            }
        }
        Ok(CompoundRule { data: parsed })
    }

    /// `CompoundRule.mayMatch`.
    pub(crate) fn may_match<W: AsRef<[i32]>>(&self, dict: &Dictionary, words: &[W]) -> bool {
        self.matches(dict, words, 0, 0, false)
    }

    /// `CompoundRule.fullyMatches`.
    pub(crate) fn fully_matches<W: AsRef<[i32]>>(&self, dict: &Dictionary, words: &[W]) -> bool {
        self.matches(dict, words, 0, 0, true)
    }

    fn matches<W: AsRef<[i32]>>(
        &self,
        dict: &Dictionary,
        words: &[W],
        pi: usize,
        mut wi: usize,
        fully: bool,
    ) -> bool {
        let data = &self.data;
        if pi >= data.len() {
            return wi >= words.len();
        }
        if wi >= words.len() && !fully {
            return true;
        }
        let flag = data[pi];
        if pi < data.len() - 1 && data[pi + 1] == u16::from(b'*') {
            let start = wi;
            while wi < words.len() && dict.has_flag_in_forms(words[wi].as_ref(), flag) {
                wi += 1;
            }
            loop {
                if self.matches(dict, words, pi + 2, wi, fully) {
                    return true;
                }
                if wi == start {
                    return false;
                }
                wi -= 1;
            }
        }
        let current = wi < words.len() && dict.has_flag_in_forms(words[wi].as_ref(), flag);
        if pi < data.len() - 1 && data[pi + 1] == u16::from(b'?') {
            if current && self.matches(dict, words, pi + 2, wi + 1, fully) {
                return true;
            }
            return self.matches(dict, words, pi + 2, wi, fully);
        }
        current && self.matches(dict, words, pi + 1, wi + 1, fully)
    }
}

/// `CheckCompoundPattern`: a `CHECKCOMPOUNDPATTERN` line.
#[derive(Debug, Clone)]
pub(crate) struct CheckCompoundPattern {
    end_chars: Vec<u16>,
    begin_chars: Vec<u16>,
    replacement: Option<Vec<u16>>,
    end_flags: Vec<u16>,
    begin_flags: Vec<u16>,
}

/// `CheckCompoundPattern.charsMatch`.
fn chars_match(word: &[u16], offset: isize, pattern: &[u16]) -> bool {
    let len = pattern.len() as isize;
    let wl = word.len() as isize;
    if wl - offset < len || offset < 0 || offset > wl {
        return false;
    }
    let o = offset as usize;
    &word[o..o + pattern.len()] == pattern
}

impl CheckCompoundPattern {
    fn new(unparsed: &[u16], parsing: FlagParsing) -> Result<Self, HunspellError> {
        let parts = java_split(unparsed, true);
        if parts.len() < 3 {
            return Err(HunspellError::IllegalArgument(format!(
                "Invalid pattern: {}",
                st(unparsed)
            )));
        }
        let split = |p: &[u16]| -> Result<(Vec<u16>, Vec<u16>), HunspellError> {
            match index_of(p, u16::from(b'/'), 0) {
                None => Ok((p.to_vec(), vec![])),
                Some(sep) => Ok((p[..sep].to_vec(), parsing.parse_flags(&p[sep + 1..])?)),
            }
        };
        let (end_chars, end_flags) = split(parts[1])?;
        let (begin_chars, begin_flags) = split(parts[2])?;
        Ok(CheckCompoundPattern {
            end_chars,
            begin_chars,
            replacement: if parts.len() == 3 {
                None
            } else {
                Some(parts[3].to_vec())
            },
            end_flags,
            begin_flags,
        })
    }

    fn is_non_affixed(p: &[u16]) -> bool {
        p == [u16::from(b'0')]
    }

    /// `CheckCompoundPattern.prohibitsCompounding`.
    pub(crate) fn prohibits_compounding(
        &self,
        dict: &Dictionary,
        word: &[u16],
        break_pos: usize,
        before: (&[u16], i32),
        after: (&[u16], i32),
    ) -> bool {
        let bp = break_pos as isize;
        if Self::is_non_affixed(&self.end_chars) {
            if !chars_match(word, bp - before.0.len() as isize, before.0) {
                return false;
            }
        } else if !chars_match(word, bp - self.end_chars.len() as isize, &self.end_chars) {
            return false;
        }
        if Self::is_non_affixed(&self.begin_chars) {
            if !chars_match(word, bp, after.0) {
                return false;
            }
        } else if !chars_match(word, bp, &self.begin_chars) {
            return false;
        }
        if !self.end_flags.is_empty() && !self.end_flags.iter().all(|&f| dict.has_flag(before.1, f))
        {
            return false;
        }
        if !self.begin_flags.is_empty()
            && !self.begin_flags.iter().all(|&f| dict.has_flag(after.1, f))
        {
            return false;
        }
        true
    }

    /// `CheckCompoundPattern.expandReplacement` over the word
    /// `chars[offset..offset + length]`: as Java, the expansion starts with
    /// the whole buffer before the break (`new String(word.chars, 0,
    /// word.offset + breakPos)`), earlier compound parts included.
    pub(crate) fn expand_replacement(
        &self,
        chars: &[u16],
        offset: usize,
        length: usize,
        break_pos: usize,
    ) -> Option<Vec<u16>> {
        let r = self.replacement.as_ref()?;
        let word = &chars[offset..offset + length];
        if !chars_match(word, break_pos as isize, r) {
            return None;
        }
        Some(
            [
                &chars[..offset + break_pos],
                self.end_chars.as_slice(),
                self.begin_chars.as_slice(),
                &word[break_pos + r.len()..],
            ]
            .concat(),
        )
    }

    /// `CheckCompoundPattern.endLength`.
    pub(crate) fn end_length(&self) -> usize {
        self.end_chars.len()
    }
}

/// A trie node: its children sorted by code point, the ids of the key ending
/// here.
type TrieNode = (Vec<(u32, usize)>, Option<Vec<i32>>);

/// The prefix or suffix table: a trie over code points whose nodes hold the
/// affix ids of the keys ending there (Lucene's `FST<IntsRef>`).
#[derive(Debug, Default)]
pub(crate) struct AffixTrie {
    nodes: Vec<TrieNode>,
    /// The keys and their ids in code-point order (`IntsRefFSTEnum`'s).
    entries: Vec<(Vec<u16>, Vec<i32>)>,
}

impl AffixTrie {
    fn build(affixes: &BTreeMap<Vec<u16>, Vec<i32>>) -> AffixTrie {
        let mut t = AffixTrie {
            nodes: vec![(Vec::new(), None)],
            entries: Vec::new(),
        };
        for (key, ids) in affixes {
            let mut node = 0;
            for c in char::decode_utf16(key.iter().copied()) {
                let cp = c.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from);
                node = match t.nodes[node].0.binary_search_by_key(&cp, |&(k, _)| k) {
                    Ok(i) => t.nodes[node].0[i].1,
                    Err(i) => {
                        t.nodes.push((Vec::new(), None));
                        let n = t.nodes.len() - 1;
                        t.nodes[node].0.insert(i, (cp, n));
                        n
                    }
                };
            }
            t.nodes[node].1 = Some(ids.clone());
            t.entries.push((key.clone(), ids.clone()));
        }
        let cps = |k: &[u16]| -> Vec<u32> {
            char::decode_utf16(k.iter().copied())
                .map(|c| c.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from))
                .collect()
        };
        t.entries.sort_by_cached_key(|(k, _)| cps(k));
        t
    }

    /// Every key with its affix ids, in code-point order.
    pub(crate) fn entries(&self) -> &[(Vec<u16>, Vec<i32>)] {
        &self.entries
    }

    /// The root node.
    pub(crate) fn root(&self) -> usize {
        0
    }

    /// The child of `node` by one UTF-16 unit (`Dictionary.nextArc`).
    pub(crate) fn step(&self, node: usize, unit: u16) -> Option<usize> {
        let children = &self.nodes[node].0;
        children
            .binary_search_by_key(&u32::from(unit), |&(k, _)| k)
            .ok()
            .map(|i| children[i].1)
    }

    /// The affix ids of the key ending at `node` (`arc.isFinal()`).
    pub(crate) fn finals(&self, node: usize) -> Option<&[i32]> {
        self.nodes[node].1.as_deref()
    }
}

/// `org.apache.lucene.analysis.hunspell.Dictionary`.
#[derive(Debug)]
pub struct Dictionary {
    pub(crate) prefixes: AffixTrie,
    pub(crate) suffixes: AffixTrie,
    pub(crate) breaks: Breaks,
    pub(crate) patterns: Vec<Option<AffixCondition>>,
    pub(crate) words: WordStorage,
    pub(crate) flag_lookup: FlagLookup,
    pub(crate) strip_data: Vec<u16>,
    pub(crate) strip_offsets: Vec<usize>,
    pub(crate) word_chars: Vec<u16>,
    affix_data: Vec<u16>,
    pub(crate) flag_parsing: FlagParsing,
    pub(crate) morph_data: Vec<Vec<u16>>,
    pub(crate) has_custom_morph_data: bool,
    pub(crate) ignore_case: bool,
    pub(crate) check_sharp_s: bool,
    pub(crate) complex_prefixes: bool,
    second_stage_prefix_flags: Vec<u16>,
    second_stage_suffix_flags: Vec<u16>,
    pub(crate) circumfix: u16,
    pub(crate) keepcase: u16,
    pub(crate) force_u_case: u16,
    pub(crate) needaffix: u16,
    pub(crate) forbiddenword: u16,
    pub(crate) onlyincompound: u16,
    pub(crate) compound_begin: u16,
    pub(crate) compound_middle: u16,
    pub(crate) compound_end: u16,
    pub(crate) compound_flag: u16,
    pub(crate) compound_permit: u16,
    pub(crate) compound_forbid: u16,
    pub(crate) check_compound_case: bool,
    pub(crate) check_compound_dup: bool,
    pub(crate) check_compound_rep: bool,
    pub(crate) check_compound_triple: bool,
    pub(crate) simplified_triple: bool,
    pub(crate) compound_min: i32,
    pub(crate) compound_max: i32,
    pub(crate) compound_rules: Option<Vec<CompoundRule>>,
    pub(crate) check_compound_patterns: Vec<CheckCompoundPattern>,
    ignore: Option<Vec<u16>>,
    pub(crate) try_chars: Vec<u16>,
    pub(crate) neighbor_key_groups: Vec<Vec<u16>>,
    pub(crate) enable_split_suggestions: bool,
    pub(crate) rep_table: Vec<RepEntry>,
    pub(crate) map_table: Vec<Vec<Vec<u16>>>,
    pub(crate) max_diff: i32,
    pub(crate) max_ngram_suggestions: i32,
    pub(crate) only_max_diff: bool,
    pub(crate) no_suggest: u16,
    pub(crate) sub_standard: u16,
    pub(crate) iconv: Option<ConvTable>,
    pub(crate) oconv: Option<ConvTable>,
    pub(crate) full_strip: bool,
    language: Option<Vec<u16>>,
    alternate_casing: bool,
}

/// One `.dic` entry as `lookupEntries` reports it (`DictEntry`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictEntry {
    /// `getStem()`.
    pub stem: String,
    /// `getFlags()`: the flags in the `.dic` encoding (possibly reordered).
    pub flags: String,
    /// `getMorphologicalData()`: sorted `kk:vvv` fields, space-separated.
    pub morphological_data: String,
}

impl DictEntry {
    /// `getMorphologicalValues(key)`.
    pub fn morphological_values(&self, key: &str) -> Vec<String> {
        if self.morphological_data.is_empty() || !self.morphological_data.contains(key) {
            return vec![];
        }
        self.morphological_data
            .split(' ')
            .filter(|s| s.starts_with(key))
            .map(|s| s[key.len()..].to_string())
            .collect()
    }
}

impl std::fmt::Display for DictEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stem)?;
        if !self.flags.is_empty() {
            write!(f, "/{}", self.flags)?;
        }
        if !self.morphological_data.is_empty() {
            write!(f, " {}", self.morphological_data)?;
        }
        Ok(())
    }
}

/// The state `readAffixFile` builds before the dictionary is complete.
struct Builder {
    d: Dictionary,
    aliases: Option<Aliases>,
    alias_count: usize,
    morph_aliases: Option<Aliases>,
    morph_alias_count: usize,
    decoder: Charset,
}

/// `Dictionary.aliases`/`morphAliases`: Java's `new String[count]`, filled
/// line by line. The announced count is a bound, not a size: the port
/// reserves at most [`MAX_ALIAS_RESERVE`] slots, so a hostile `AF 2000000000`
/// costs nothing until that many lines exist.
struct Aliases {
    /// The header's count (`array.length`).
    declared: usize,
    /// The lines read so far.
    values: Vec<Vec<u16>>,
}

/// The most alias slots reserved before their lines are read.
const MAX_ALIAS_RESERVE: usize = 1024;

impl Aliases {
    /// `new String[count]`: Java's `NegativeArraySizeException` for a
    /// negative count.
    fn new(count: i32) -> Result<Aliases, HunspellError> {
        let declared = usize::try_from(count)
            .map_err(|_| HunspellError::NegativeArraySize(count.to_string()))?;
        Ok(Aliases {
            declared,
            values: Vec::with_capacity(declared.min(MAX_ALIAS_RESERVE)),
        })
    }

    /// `array[count++] = value`: Java's `ArrayIndexOutOfBoundsException`
    /// past the announced count.
    fn push(&mut self, value: Vec<u16>) -> Result<(), HunspellError> {
        if self.values.len() >= self.declared {
            return Err(HunspellError::IndexOutOfBounds(format!(
                "Index {} out of bounds for length {}",
                self.values.len(),
                self.declared
            )));
        }
        self.values.push(value);
        Ok(())
    }

    /// `array[id - 1]`, `None` outside the lines read (Java: out of the
    /// array, or a `null` slot inside it).
    fn get(&self, id: i32) -> Option<&Vec<u16>> {
        let i = usize::try_from(id.checked_sub(1)?).ok()?;
        self.values.get(i)
    }
}

impl Dictionary {
    /// `new Dictionary(affix, dictionaries, ignoreCase, SortingStrategy.inMemory())`
    /// over the files' bytes.
    pub fn new(
        affix: &[u8],
        dictionaries: &[&[u8]],
        ignore_case: bool,
    ) -> Result<Dictionary, HunspellError> {
        let mut b = Builder {
            d: Dictionary::empty(ignore_case),
            aliases: None,
            alias_count: 0,
            morph_aliases: None,
            morph_alias_count: 0,
            decoder: Charset::Iso8859_1,
        };
        let (affix, stream_charset) = match affix.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
            Some(rest) => (rest, Charset::Utf8),
            None => (affix, Charset::Iso8859_1),
        };
        let prologue = &affix[..affix.len().min(MAX_PROLOGUE_SCAN_WINDOW - 1)];
        b.read_config(stream_charset.decode(prologue)?)?;
        let mut flags = FlagEnumerator::new();
        b.read_affix_file(b.decoder.decode(affix)?, &mut flags)?;
        let mut entries = Vec::new();
        for dic in dictionaries {
            b.merge_dictionary(dic, &mut entries)?;
        }
        entries.sort();
        b.read_sorted_dictionaries(&mut flags, entries)?;
        b.d.flag_lookup = flags.finish();
        Ok(b.d)
    }

    fn empty(ignore_case: bool) -> Dictionary {
        Dictionary {
            prefixes: AffixTrie::default(),
            suffixes: AffixTrie::default(),
            breaks: Breaks::default_breaks(),
            patterns: Vec::new(),
            words: WordStorageBuilder::new(0, 1.0, false, &mut FlagEnumerator::new(), vec![])
                .finish()
                .expect("an empty word storage"),
            flag_lookup: FlagLookup::default(),
            strip_data: Vec::new(),
            strip_offsets: Vec::new(),
            word_chars: Vec::new(),
            affix_data: Vec::new(),
            flag_parsing: FlagParsing::Simple,
            morph_data: vec![Vec::new()],
            has_custom_morph_data: false,
            ignore_case,
            check_sharp_s: false,
            complex_prefixes: false,
            second_stage_prefix_flags: Vec::new(),
            second_stage_suffix_flags: Vec::new(),
            circumfix: FLAG_UNSET,
            keepcase: FLAG_UNSET,
            force_u_case: FLAG_UNSET,
            needaffix: FLAG_UNSET,
            forbiddenword: FLAG_UNSET,
            onlyincompound: FLAG_UNSET,
            compound_begin: FLAG_UNSET,
            compound_middle: FLAG_UNSET,
            compound_end: FLAG_UNSET,
            compound_flag: FLAG_UNSET,
            compound_permit: FLAG_UNSET,
            compound_forbid: FLAG_UNSET,
            check_compound_case: false,
            check_compound_dup: false,
            check_compound_rep: false,
            check_compound_triple: false,
            simplified_triple: false,
            compound_min: 3,
            compound_max: i32::MAX,
            compound_rules: None,
            check_compound_patterns: Vec::new(),
            ignore: None,
            try_chars: Vec::new(),
            neighbor_key_groups: vec![u("qwertyuiop"), u("asdfghjkl"), u("zxcvbnm")],
            enable_split_suggestions: true,
            rep_table: Vec::new(),
            map_table: Vec::new(),
            max_diff: 5,
            max_ngram_suggestions: 4,
            only_max_diff: false,
            no_suggest: FLAG_UNSET,
            sub_standard: FLAG_UNSET,
            iconv: None,
            oconv: None,
            full_strip: false,
            language: None,
            alternate_casing: false,
        }
    }

    /// `formStep()`: forms hold a morph-data id after each flag-set id when
    /// the dictionary has custom morphological data.
    pub(crate) fn form_step(&self) -> usize {
        if self.has_custom_morph_data {
            2
        } else {
            1
        }
    }

    /// `lookupWord`.
    pub(crate) fn lookup_word(&self, word: &[u16]) -> Option<&[i32]> {
        self.words.lookup_word(word)
    }

    /// `affixData(affixIndex, offset)`.
    pub(crate) fn affix_data(&self, affix: i32, offset: usize) -> u16 {
        self.affix_data[affix as usize * 4 + offset]
    }

    /// `isCrossProduct`.
    pub(crate) fn is_cross_product(&self, affix: i32) -> bool {
        self.affix_data(affix, AFFIX_CONDITION) & 1 == 1
    }

    /// `getAffixCondition`.
    pub(crate) fn get_affix_condition(&self, affix: i32) -> usize {
        usize::from(self.affix_data(affix, AFFIX_CONDITION) >> 1)
    }

    /// `hasFlag(int entryId, char flag)`.
    pub(crate) fn has_flag(&self, entry_id: i32, flag: u16) -> bool {
        self.flag_lookup.has_flag(entry_id, flag)
    }

    /// `hasFlag(IntsRef forms, char flag)`.
    pub(crate) fn has_flag_in_forms(&self, forms: &[i32], flag: u16) -> bool {
        forms
            .iter()
            .step_by(self.form_step())
            .any(|&id| self.has_flag(id, flag))
    }

    /// `isFlagAppendedByAffix`.
    pub(crate) fn is_flag_appended_by_affix(&self, affix_id: i32, flag: u16) -> bool {
        if affix_id < 0 || flag == FLAG_UNSET {
            return false;
        }
        let append = self.affix_data(affix_id, AFFIX_APPEND);
        self.has_flag(i32::from(append), flag)
    }

    /// `isSecondStagePrefix`.
    pub(crate) fn is_second_stage_prefix(&self, flag: u16) -> bool {
        self.second_stage_prefix_flags.binary_search(&flag).is_ok()
    }

    /// `isSecondStageSuffix`.
    pub(crate) fn is_second_stage_suffix(&self, flag: u16) -> bool {
        self.second_stage_suffix_flags.binary_search(&flag).is_ok()
    }

    /// `caseFold(char)`.
    pub(crate) fn case_fold(&self, c: u16) -> u16 {
        if self.alternate_casing {
            if c == u16::from(b'I') {
                return 0x0131;
            }
            if c == 0x0130 {
                return u16::from(b'i');
            }
        }
        to_lower(c)
    }

    /// `toLowerCase(String)`.
    pub(crate) fn to_lower_case(&self, word: &[u16]) -> Vec<u16> {
        word.iter().map(|&c| self.case_fold(c)).collect()
    }

    /// `toTitleCase(String)`.
    pub(crate) fn to_title_case(&self, word: &[u16]) -> Vec<u16> {
        let mut out = Vec::with_capacity(word.len());
        if let Some((&first, rest)) = word.split_first() {
            out.push(to_upper(first));
            out.extend(rest.iter().map(|&c| self.case_fold(c)));
        }
        out
    }

    /// `isDotICaseChangeDisallowed`.
    pub(crate) fn is_dot_i_case_change_disallowed(&self, word: &[u16]) -> bool {
        word.first() == Some(&0x0130) && !self.alternate_casing
    }

    /// `hasLanguage(langCodes...)`.
    pub(crate) fn has_language(&self, codes: &[&str]) -> bool {
        let Some(lang) = &self.language else {
            return false;
        };
        let code = match index_of(lang, u16::from(b'_'), 0) {
            Some(us) => &lang[..us],
            None => &lang[..],
        };
        codes.iter().any(|c| u(c) == code)
    }

    /// `mayNeedInputCleaning`.
    pub(crate) fn may_need_input_cleaning(&self) -> bool {
        self.ignore_case || self.ignore.is_some() || self.iconv.is_some()
    }

    /// `needsInputCleaning`.
    pub(crate) fn needs_input_cleaning(&self, input: &[u16]) -> bool {
        if self.may_need_input_cleaning() {
            for &ch in input {
                if self
                    .ignore
                    .as_ref()
                    .is_some_and(|ig| ig.binary_search(&ch).is_ok())
                    || self.ignore_case && self.case_fold(ch) != ch
                    || self
                        .iconv
                        .as_ref()
                        .is_some_and(|c| c.might_replace_char(ch))
                {
                    return true;
                }
            }
        }
        false
    }

    /// `cleanInput`.
    pub(crate) fn clean_input(&self, input: &[u16]) -> Vec<u16> {
        let mut out = Vec::with_capacity(input.len());
        for &ch in input {
            if self
                .ignore
                .as_ref()
                .is_some_and(|ig| ig.binary_search(&ch).is_ok())
            {
                continue;
            }
            out.push(if self.ignore_case && self.iconv.is_none() {
                self.case_fold(ch)
            } else {
                ch
            });
        }
        if let Some(iconv) = &self.iconv {
            iconv.apply_mappings(&mut out);
            if self.ignore_case {
                for c in out.iter_mut() {
                    *c = self.case_fold(*c);
                }
            }
        }
        out
    }

    /// `getIgnoreCase()`.
    pub fn ignore_case(&self) -> bool {
        self.ignore_case
    }

    /// `lookupEntries(root)`: the homonyms of `root`, or `None` when it is
    /// not in the dictionary.
    pub fn lookup_entries(&self, root: &str) -> Option<Vec<DictEntry>> {
        let units = u(root);
        if units.is_empty() {
            return None;
        }
        let forms = self.lookup_word(&units)?;
        let step = self.form_step();
        Some(
            forms
                .chunks(step)
                .map(|f| self.dict_entry(&units, f[0], if step == 2 { f[1] } else { 0 }))
                .collect(),
        )
    }

    /// `dictEntry(root, flagId, morphDataId)`.
    pub(crate) fn dict_entry(&self, root: &[u16], flag_id: i32, morph_data_id: i32) -> DictEntry {
        DictEntry {
            stem: st(root),
            flags: self
                .flag_parsing
                .print_flags(&self.flag_lookup.get_flags(flag_id)),
            morphological_data: if morph_data_id == 0 {
                String::new()
            } else {
                st(&self.morph_data[morph_data_id as usize])
            },
        }
    }

    /// `allNonSuggestibleFlags()`.
    fn all_non_suggestible_flags(&self) -> Vec<u16> {
        let mut set = vec![HIDDEN_FLAG];
        for c in [
            self.no_suggest,
            self.forbiddenword,
            self.onlyincompound,
            self.sub_standard,
        ] {
            if c != FLAG_UNSET && !set.contains(&c) {
                set.push(c);
            }
        }
        set.sort_unstable();
        set
    }
}

impl Builder {
    fn parse_error(&self, msg: impl Into<String>, line: usize) -> HunspellError {
        HunspellError::Parse {
            message: msg.into(),
            line,
        }
    }

    /// `splitBySpace(reader, line, min, max)`.
    fn split_by_space<'l>(
        &self,
        reader: &LineReader,
        line: &'l [u16],
        min: usize,
        max: usize,
    ) -> Result<Vec<&'l [u16]>, HunspellError> {
        let parts = java_split(line, true);
        if parts.len() < min || parts.len() > max && !parts[max].starts_with(&[u16::from(b'#')]) {
            return Err(
                self.parse_error(format!("Invalid syntax: {}", st(line)), reader.line_number)
            );
        }
        Ok(parts)
    }

    fn single_argument(
        &self,
        reader: &LineReader,
        line: &[u16],
    ) -> Result<Vec<u16>, HunspellError> {
        Ok(self.split_by_space(reader, line, 2, 2)?[1].to_vec())
    }

    fn first_argument(&self, reader: &LineReader, line: &[u16]) -> Result<Vec<u16>, HunspellError> {
        Ok(self.split_by_space(reader, line, 2, usize::MAX)?[1].to_vec())
    }

    fn parse_num(&self, reader: &LineReader, line: &[u16]) -> Result<i32, HunspellError> {
        parse_int(self.split_by_space(reader, line, 2, usize::MAX)?[1])
    }

    fn next_line(&self, reader: &mut LineReader) -> Result<Vec<u16>, HunspellError> {
        reader
            .read_line()
            .ok_or_else(|| self.parse_error("Unexpected end of file", reader.line_number))
    }

    fn parse_flag(&self, reader: &LineReader, line: &[u16]) -> Result<u16, HunspellError> {
        let arg = self.single_argument(reader, line)?;
        self.d.flag_parsing.parse_flag(&arg)
    }

    /// `readConfig`: `SET` and `FLAG` from the prologue.
    fn read_config(&mut self, prologue: Vec<u16>) -> Result<(), HunspellError> {
        let mut reader = LineReader::new(prologue);
        let mut flag_line = None;
        let (mut charset_found, mut flag_found) = (false, false);
        while let Some(line) = reader.read_line() {
            if java_is_blank(&line) {
                continue;
            }
            let first = java_split(&line, false)
                .first()
                .map(|p| p.to_vec())
                .unwrap_or_default();
            if first == u("SET") {
                self.decoder = Charset::for_name(&self.single_argument(&reader, &line)?)?;
                charset_found = true;
            } else if first == u("FLAG") {
                flag_line = Some(line);
                flag_found = true;
            } else {
                continue;
            }
            if charset_found && flag_found {
                break;
            }
        }
        if let Some(line) = flag_line {
            self.d.flag_parsing = flag_parsing_strategy(&line, self.decoder)?;
        }
        Ok(())
    }

    /// `readAffixFile`.
    fn read_affix_file(
        &mut self,
        text: Vec<u16>,
        flags: &mut FlagEnumerator,
    ) -> Result<(), HunspellError> {
        let mut prefixes: BTreeMap<Vec<u16>, Vec<i32>> = BTreeMap::new();
        let mut suffixes: BTreeMap<Vec<u16>, Vec<i32>> = BTreeMap::new();
        let mut prefix_cont: Vec<u16> = Vec::new();
        let mut suffix_cont: Vec<u16> = Vec::new();
        let mut seen_patterns: HashMap<String, usize> = HashMap::new();
        seen_patterns.insert(ALWAYS_TRUE_KEY.to_string(), 0);
        self.d.patterns.push(None);
        let mut seen_strips: Vec<Vec<u16>> = vec![Vec::new()];
        let mut strip_index: HashMap<Vec<u16>, usize> = HashMap::from([(Vec::new(), 0)]);
        let mut reader = LineReader::new(text);
        while let Some(raw) = reader.read_line() {
            let mut line: &[u16] = &raw;
            if reader.line_number == 1 && line.first() == Some(&0xFEFF) {
                line = &line[1..];
            }
            let line = java_trim(line).to_vec();
            if line.is_empty() {
                continue;
            }
            let first = st(java_split(&line, false).first().copied().unwrap_or(&[]));
            match first.as_str() {
                "AF" => self.parse_alias(&line)?,
                "AM" => self.parse_morph_alias(&line)?,
                "PFX" | "SFX" => {
                    let kind = if first == "PFX" {
                        AffixKind::Prefix
                    } else {
                        AffixKind::Suffix
                    };
                    let (table, cont) = if kind == AffixKind::Prefix {
                        (&mut prefixes, &mut prefix_cont)
                    } else {
                        (&mut suffixes, &mut suffix_cont)
                    };
                    self.parse_affix(
                        table,
                        cont,
                        &line,
                        &mut reader,
                        kind,
                        &mut seen_patterns,
                        (&mut seen_strips, &mut strip_index),
                        flags,
                    )?;
                }
                _ if line == u("COMPLEXPREFIXES") => self.d.complex_prefixes = true,
                "CIRCUMFIX" => self.d.circumfix = self.parse_flag(&reader, &line)?,
                "KEEPCASE" => self.d.keepcase = self.parse_flag(&reader, &line)?,
                "FORCEUCASE" => self.d.force_u_case = self.parse_flag(&reader, &line)?,
                "NEEDAFFIX" | "PSEUDOROOT" => self.d.needaffix = self.parse_flag(&reader, &line)?,
                "ONLYINCOMPOUND" => self.d.onlyincompound = self.parse_flag(&reader, &line)?,
                "CHECKSHARPS" => self.d.check_sharp_s = true,
                "IGNORE" => {
                    let mut ig = self.single_argument(&reader, &line)?;
                    ig.sort_unstable();
                    self.d.ignore = Some(ig);
                }
                "ICONV" | "OCONV" => {
                    let num = self.parse_num(&reader, &line)?;
                    let table = self.parse_conversions(&mut reader, num)?;
                    if first == "ICONV" {
                        self.d.iconv = Some(table);
                    } else {
                        self.d.oconv = Some(table);
                    }
                }
                "FULLSTRIP" => self.d.full_strip = true,
                "LANG" => {
                    self.d.language = Some(self.single_argument(&reader, &line)?);
                    self.d.alternate_casing = self.d.has_language(&["tr", "az"]);
                }
                "BREAK" => self.d.breaks = self.parse_breaks(&mut reader, &line)?,
                "WORDCHARS" => self.d.word_chars = self.first_argument(&reader, &line)?,
                "TRY" => self.d.try_chars = self.first_argument(&reader, &line)?,
                "REP" => {
                    let count = self.parse_num(&reader, &line)?;
                    for _ in 0..count.max(0) {
                        let l = self.next_line(&mut reader)?;
                        let parts = self.split_by_space(&reader, &l, 3, usize::MAX)?;
                        self.d.rep_table.push(RepEntry::new(parts[1], parts[2]));
                    }
                }
                "MAP" => {
                    let count = self.parse_num(&reader, &line)?;
                    for _ in 0..count.max(0) {
                        let l = self.next_line(&mut reader)?;
                        let entry = self.parse_map_entry(&reader, &l)?;
                        self.d.map_table.push(entry);
                    }
                }
                "KEY" => {
                    let arg = self.single_argument(&reader, &line)?;
                    let mut groups: Vec<Vec<u16>> = arg
                        .split(|&c| c == u16::from(b'|'))
                        .map(|g| g.to_vec())
                        .collect();
                    while groups.len() > 1 && groups.last().is_some_and(Vec::is_empty) {
                        groups.pop();
                    }
                    self.d.neighbor_key_groups = groups;
                }
                "NOSPLITSUGS" => self.d.enable_split_suggestions = false,
                "MAXNGRAMSUGS" => {
                    self.d.max_ngram_suggestions =
                        parse_int(&self.single_argument(&reader, &line)?)?
                }
                "MAXDIFF" => {
                    let i = parse_int(&self.single_argument(&reader, &line)?)?;
                    if !(0..=10).contains(&i) {
                        return Err(self.parse_error(
                            "MAXDIFF should be between 0 and 10",
                            reader.line_number,
                        ));
                    }
                    self.d.max_diff = i;
                }
                "ONLYMAXDIFF" => self.d.only_max_diff = true,
                "FORBIDDENWORD" => self.d.forbiddenword = self.parse_flag(&reader, &line)?,
                "NOSUGGEST" => self.d.no_suggest = self.parse_flag(&reader, &line)?,
                "SUBSTANDARD" => self.d.sub_standard = self.parse_flag(&reader, &line)?,
                "COMPOUNDMIN" => self.d.compound_min = self.parse_num(&reader, &line)?.max(1),
                "COMPOUNDWORDMAX" => self.d.compound_max = self.parse_num(&reader, &line)?.max(1),
                "COMPOUNDRULE" => {
                    let num = self.parse_num(&reader, &line)?;
                    let mut rules = Vec::new();
                    for _ in 0..num.max(0) {
                        let l = self.next_line(&mut reader)?;
                        let arg = self.single_argument(&reader, &l)?;
                        rules.push(CompoundRule::new(&arg, self.d.flag_parsing)?);
                    }
                    self.d.compound_rules = Some(rules);
                }
                "COMPOUNDFLAG" => self.d.compound_flag = self.parse_flag(&reader, &line)?,
                "COMPOUNDBEGIN" => self.d.compound_begin = self.parse_flag(&reader, &line)?,
                "COMPOUNDMIDDLE" => self.d.compound_middle = self.parse_flag(&reader, &line)?,
                "COMPOUNDEND" => self.d.compound_end = self.parse_flag(&reader, &line)?,
                "COMPOUNDPERMITFLAG" => self.d.compound_permit = self.parse_flag(&reader, &line)?,
                "COMPOUNDFORBIDFLAG" => self.d.compound_forbid = self.parse_flag(&reader, &line)?,
                "CHECKCOMPOUNDCASE" => self.d.check_compound_case = true,
                "CHECKCOMPOUNDDUP" => self.d.check_compound_dup = true,
                "CHECKCOMPOUNDREP" => self.d.check_compound_rep = true,
                "CHECKCOMPOUNDTRIPLE" => self.d.check_compound_triple = true,
                "SIMPLIFIEDTRIPLE" => self.d.simplified_triple = true,
                "CHECKCOMPOUNDPATTERN" => {
                    let count = self.parse_num(&reader, &line)?;
                    for _ in 0..count.max(0) {
                        let l = self.next_line(&mut reader)?;
                        let p = CheckCompoundPattern::new(&l, self.d.flag_parsing)?;
                        self.d.check_compound_patterns.push(p);
                    }
                }
                "SET" => {
                    let cs = Charset::for_name(&self.single_argument(&reader, &line)?)?;
                    if cs.java_charset() != self.decoder.java_charset() {
                        return Err(self.critical_directive("SET", &reader));
                    }
                }
                "FLAG" => {
                    let strategy = flag_parsing_strategy(&line, self.decoder)?;
                    if strategy != self.d.flag_parsing {
                        return Err(self.critical_directive("FLAG", &reader));
                    }
                }
                _ => {}
            }
        }
        self.d.prefixes = AffixTrie::build(&prefixes);
        self.d.suffixes = AffixTrie::build(&suffixes);
        prefix_cont.sort_unstable();
        prefix_cont.dedup();
        suffix_cont.sort_unstable();
        suffix_cont.dedup();
        self.d.second_stage_prefix_flags = prefix_cont;
        self.d.second_stage_suffix_flags = suffix_cont;
        let mut offset = 0;
        for strip in &seen_strips {
            self.d.strip_offsets.push(offset);
            self.d.strip_data.extend_from_slice(strip);
            offset += strip.len();
        }
        self.d.strip_offsets.push(offset);
        Ok(())
    }

    fn critical_directive(&self, directive: &str, reader: &LineReader) -> HunspellError {
        self.parse_error(
            format!(
                "{directive} directive should occur at most once, and in the first {MAX_PROLOGUE_SCAN_WINDOW} bytes of the *.aff file"
            ),
            reader.line_number,
        )
    }

    /// `parseMapEntry`.
    fn parse_map_entry(
        &self,
        reader: &LineReader,
        line: &[u16],
    ) -> Result<Vec<Vec<u16>>, HunspellError> {
        let unparsed = self.first_argument(reader, line)?;
        let mut entry = Vec::new();
        let mut j = 0;
        while j < unparsed.len() {
            if unparsed[j] == u16::from(b'(') {
                let Some(closing) = index_of(&unparsed, u16::from(b')'), j) else {
                    return Err(self.parse_error(
                        format!("Unclosed parenthesis: {}", st(line)),
                        reader.line_number,
                    ));
                };
                entry.push(unparsed[j + 1..closing].to_vec());
                j = closing;
            } else {
                entry.push(vec![unparsed[j]]);
            }
            j += 1;
        }
        Ok(entry)
    }

    /// `parseBreaks`.
    fn parse_breaks(&self, reader: &mut LineReader, line: &[u16]) -> Result<Breaks, HunspellError> {
        let (mut starting, mut ending, mut middle) = (Vec::new(), Vec::new(), Vec::new());
        let num = self.parse_num(reader, line)?;
        let add = |set: &mut Vec<Vec<u16>>, s: Vec<u16>| {
            if !set.contains(&s) {
                set.push(s);
            }
        };
        for _ in 0..num.max(0) {
            let l = self.next_line(reader)?;
            let b = self.single_argument(reader, &l)?;
            if b.first() == Some(&u16::from(b'^')) {
                add(&mut starting, b[1..].to_vec());
            } else if b.last() == Some(&u16::from(b'$')) {
                add(&mut ending, b[..b.len() - 1].to_vec());
            } else {
                add(&mut middle, b);
            }
        }
        Ok(Breaks {
            starting,
            ending,
            middle,
        })
    }

    /// `parseConversions`.
    fn parse_conversions(
        &self,
        reader: &mut LineReader,
        num: i32,
    ) -> Result<ConvTable, HunspellError> {
        let mut mappings = BTreeMap::new();
        for _ in 0..num.max(0) {
            let l = self.next_line(reader)?;
            let parts = self.split_by_space(reader, &l, 3, 3)?;
            if mappings
                .insert(parts[1].to_vec(), parts[2].to_vec())
                .is_some()
            {
                return Err(HunspellError::IllegalState(format!(
                    "duplicate mapping specified for: {}",
                    st(parts[1])
                )));
            }
        }
        Ok(ConvTable::new(&mappings))
    }

    /// `parseAlias` (`AF`).
    fn parse_alias(&mut self, line: &[u16]) -> Result<(), HunspellError> {
        let args = java_split(line, true);
        match &mut self.aliases {
            None => {
                let count = parse_int(args.get(1).copied().unwrap_or(&[]))?;
                self.aliases = Some(Aliases::new(count)?);
            }
            Some(aliases) => {
                // an alias can map to no flags
                let value = if args.len() == 1 {
                    Vec::new()
                } else {
                    args[1].to_vec()
                };
                aliases.push(value)?;
                self.alias_count += 1;
            }
        }
        Ok(())
    }

    /// `getAliasValue`. An id inside the announced count but past the lines
    /// read is Java's `null` slot (a `NullPointerException` later); the port
    /// reports it as a bad alias number.
    fn alias_value(&self, id: i32) -> Result<Vec<u16>, HunspellError> {
        self.aliases
            .as_ref()
            .and_then(|a| a.get(id))
            .cloned()
            .ok_or_else(|| HunspellError::IllegalArgument(format!("Bad flag alias number:{id}")))
    }

    /// `parseMorphAlias` (`AM`).
    fn parse_morph_alias(&mut self, line: &[u16]) -> Result<(), HunspellError> {
        match &mut self.morph_aliases {
            None => {
                let count = parse_int(line.get(3..).unwrap_or(&[]))?;
                self.morph_aliases = Some(Aliases::new(count)?);
            }
            Some(aliases) => {
                aliases.push(line[2..].to_vec())?; // leave the space
                self.morph_alias_count += 1;
            }
        }
        Ok(())
    }

    /// `parseAffix`.
    #[allow(clippy::too_many_arguments)]
    fn parse_affix(
        &mut self,
        affixes: &mut BTreeMap<Vec<u16>, Vec<i32>>,
        second_stage: &mut Vec<u16>,
        header: &[u16],
        reader: &mut LineReader,
        kind: AffixKind,
        seen_patterns: &mut HashMap<String, usize>,
        (seen_strips, strip_index): (&mut Vec<Vec<u16>>, &mut HashMap<Vec<u16>, usize>),
        flags: &mut FlagEnumerator,
    ) -> Result<(), HunspellError> {
        let args = java_split(header, true);
        let cross_product = args.get(2).copied() == Some(&[u16::from(b'Y')][..]);
        let num_lines = match args.get(3).map(|a| parse_int(a)) {
            Some(Ok(n)) => n,
            _ => {
                return Err(self.parse_error(
                    format!("Affix rule header expected; got {}", st(header)),
                    reader.line_number,
                ))
            }
        };
        for _ in 0..num_lines.max(0) {
            let Some(line) = reader.read_line() else {
                return Err(self.parse_error(
                    format!("Premature end of rules for {}", st(header)),
                    reader.line_number,
                ));
            };
            let rule = self.split_by_space(reader, &line, 4, usize::MAX)?;
            if rule[1] != args[1] {
                return Err(self.parse_error(
                    format!(
                        "Affix rule mismatch. Header: {}; rule: {}",
                        st(header),
                        st(&line)
                    ),
                    reader.line_number,
                ));
            }
            let flag = self.d.flag_parsing.parse_flag(rule[1])?;
            let strip: Vec<u16> = if rule[2] == [u16::from(b'0')] {
                vec![]
            } else {
                rule[2].to_vec()
            };
            let mut affix_arg = rule[3].to_vec();
            let mut append_flags = Vec::new();
            if let Some(sep) = affix_arg.iter().rposition(|&c| c == u16::from(b'/')) {
                let mut flag_part = affix_arg[sep + 1..].to_vec();
                affix_arg.truncate(sep);
                if self.alias_count > 0 {
                    flag_part = self.alias_value(parse_int(&flag_part)?)?;
                }
                append_flags = self.d.flag_parsing.parse_flags(&flag_part)?;
                for &f in &append_flags {
                    if !second_stage.contains(&f) {
                        second_stage.push(f);
                    }
                }
            }
            if affix_arg == [u16::from(b'0')] {
                affix_arg.clear();
            }
            let condition = if rule.len() > 4 {
                rule[4].to_vec()
            } else {
                u(".")
            };
            let key = unique_key(kind, &strip, &condition);
            let pattern_index = match seen_patterns.get(&key) {
                Some(&i) => i,
                None => {
                    let i = self.d.patterns.len();
                    if i > i16::MAX as usize {
                        return Err(HunspellError::Unsupported(
                            "Too many patterns, please report this to dev@lucene.apache.org".into(),
                        ));
                    }
                    seen_patterns.insert(key, i);
                    self.d.patterns.push(Some(AffixCondition::compile(
                        kind, &strip, &condition, &line,
                    )?));
                    i
                }
            };
            let strip_ord = match strip_index.get(&strip) {
                Some(&o) => o,
                None => {
                    let o = seen_strips.len();
                    if o > usize::from(u16::MAX) {
                        return Err(HunspellError::Unsupported(
                            "Too many unique strips, please report this to dev@lucene.apache.org"
                                .into(),
                        ));
                    }
                    strip_index.insert(strip.clone(), o);
                    seen_strips.push(strip);
                    o
                }
            };
            let append_ord = flags.add(&mut append_flags)?;
            if append_ord > i32::from(i16::MAX) {
                return Err(HunspellError::Unsupported(
                    "Too many unique append flags, please report this to dev@lucene.apache.org"
                        .into(),
                ));
            }
            let current_affix = (self.d.affix_data.len() / 4) as i32;
            let pattern_ord = (pattern_index << 1) | usize::from(cross_product);
            self.d.affix_data.extend_from_slice(&[
                flag,
                strip_ord as u16,
                pattern_ord as u16,
                append_ord as u16,
            ]);
            if self.d.needs_input_cleaning(&affix_arg) {
                affix_arg = self.d.clean_input(&affix_arg);
            }
            if kind == AffixKind::Suffix {
                affix_arg = reverse_code_points(&affix_arg);
            }
            affixes.entry(affix_arg).or_default().push(current_affix);
        }
        Ok(())
    }

    /// `unescapeEntry`.
    fn unescape_entry(entry: &[u16]) -> Vec<u16> {
        let mut sb = Vec::with_capacity(entry.len() + 1);
        let end = morph_boundary(entry);
        let mut i = 0;
        while i < end {
            let ch = entry[i];
            if ch == u16::from(b'\\') && i + 1 < entry.len() {
                sb.push(entry[i + 1]);
                i += 1;
            } else if ch == u16::from(b'/') && i > 0 {
                sb.push(FLAG_SEPARATOR);
            } else if ch != FLAG_SEPARATOR && ch != MORPH_SEPARATOR {
                sb.push(ch);
            }
            i += 1;
        }
        sb.push(MORPH_SEPARATOR);
        if end < entry.len() {
            sb.extend(
                entry[end..]
                    .iter()
                    .filter(|&&c| c != FLAG_SEPARATOR && c != MORPH_SEPARATOR),
            );
        }
        sb
    }

    /// `mergeDictionaries` for one `.dic` file.
    fn merge_dictionary(
        &mut self,
        dic: &[u8],
        acc: &mut Vec<Vec<u16>>,
    ) -> Result<(), HunspellError> {
        let mut lines = LineReader::new(self.decoder.decode(dic)?);
        lines.read_line(); // the (approximate) entry count
        while let Some(line) = lines.read_line() {
            if line.is_empty() || line[0] == u16::from(b'#') || line[0] == u16::from(b'\t') {
                continue;
            }
            let line = Self::unescape_entry(&line);
            if !self.d.has_custom_morph_data {
                if let Some(start) = index_of(&line, MORPH_SEPARATOR, 0) {
                    let data = line[start + 1..].to_vec();
                    self.d.has_custom_morph_data = self
                        .split_morph_data(&data)?
                        .iter()
                        .any(|s| !s.starts_with(&u("ph:")));
                }
            }
            self.write_normalized_word_entry(&line, acc);
        }
        Ok(())
    }

    /// `writeNormalizedWordEntry`.
    fn write_normalized_word_entry(&self, line: &[u16], acc: &mut Vec<Vec<u16>>) {
        let flag_sep = index_of(line, FLAG_SEPARATOR, 0);
        let morph_sep = index_of(line, MORPH_SEPARATOR, 0).unwrap_or(line.len());
        let sep = flag_sep.unwrap_or(morph_sep);
        if sep == 0 {
            return;
        }
        let before = &line[..sep];
        let written = if self.d.needs_input_cleaning(before) {
            [self.d.clean_input(before).as_slice(), &line[sep..]].concat()
        } else {
            line.to_vec()
        };
        let sep = written.len() - (line.len() - sep);
        acc.push(written.clone());
        if sep == 0 {
            return;
        }
        let case = WordCase::case_of(&written[..sep]);
        if case == WordCase::Mixed || case == WordCase::Upper && flag_sep.is_some_and(|f| f > 0) {
            // `addHiddenCapitalizedWord`.
            let word = &written[..sep];
            let after = &written[sep..];
            let mut hidden = vec![to_upper(word[0])];
            hidden.extend(word[1..].iter().map(|&c| self.d.case_fold(c)));
            hidden.push(FLAG_SEPARATOR);
            hidden.push(HIDDEN_FLAG);
            hidden.extend_from_slice(&after[usize::from(after[0] == FLAG_SEPARATOR)..]);
            acc.push(hidden);
        }
    }

    /// `splitMorphData`. A numeric alias outside the announced `AM` count
    /// is Java's `ArrayIndexOutOfBoundsException`; one inside it but past the
    /// lines given (Java's `null` slot, a `NullPointerException`) is reported
    /// the same way.
    fn split_morph_data(&self, data: &[u16]) -> Result<Vec<Vec<u16>>, HunspellError> {
        let mut data = data.to_vec();
        if let Some(aliases) = self
            .morph_aliases
            .as_ref()
            .filter(|_| self.morph_alias_count > 0)
        {
            if let Ok(alias) = parse_int(java_trim(&data)) {
                data = aliases.get(alias).cloned().ok_or_else(|| {
                    HunspellError::IndexOutOfBounds(format!(
                        "Index {} out of bounds for length {}",
                        alias.wrapping_sub(1),
                        aliases.declared
                    ))
                })?;
            }
        }
        if java_is_blank(&data) {
            return Ok(vec![]);
        }
        let mut result = Vec::new();
        let mut start = 0;
        for i in 0..=data.len() {
            if i == data.len() || jc::is_whitespace(u32::from(data[i])) {
                if i >= start + 4
                    && jc::is_letter(u32::from(data[start]))
                    && jc::is_letter(u32::from(data[start + 1]))
                    && data[start + 2] == u16::from(b':')
                {
                    result.push(data[start..i].to_vec());
                }
                start = i + 1;
            }
        }
        Ok(result)
    }

    /// `readSortedDictionaries`.
    fn read_sorted_dictionaries(
        &mut self,
        flags: &mut FlagEnumerator,
        sorted: Vec<Vec<u16>>,
    ) -> Result<(), HunspellError> {
        let mut morph_indices: HashMap<Vec<u16>, i32> = HashMap::new();
        let no_suggest = self.d.all_non_suggestible_flags();
        let has_morph = self.d.has_custom_morph_data;
        let word_count = sorted.len();
        let mut pending: Vec<(Vec<u16>, Vec<u16>, i32)> = Vec::with_capacity(sorted.len());
        for line in sorted {
            let end = index_of(&line, MORPH_SEPARATOR, 0).unwrap_or(line.len());
            let (entry, word_form) = match index_of(&line, FLAG_SEPARATOR, 0) {
                None => (line[..end].to_vec(), Vec::new()),
                Some(flag_sep) => {
                    let hidden = line.get(flag_sep + 1) == Some(&HIDDEN_FLAG);
                    let from = (flag_sep + if hidden { 2 } else { 1 }).min(end);
                    let mut flag_part = java_strip(&line[from..end]).to_vec();
                    if self.alias_count > 0 && !flag_part.is_empty() {
                        flag_part = self.alias_value(parse_int(&flag_part)?)?;
                    }
                    let mut form = self.d.flag_parsing.parse_flags(&flag_part)?;
                    if hidden {
                        form.push(HIDDEN_FLAG);
                    }
                    (line[..flag_sep].to_vec(), form)
                }
            };
            if entry.is_empty() {
                continue;
            }
            let mut morph_id = 0;
            if end + 1 < line.len() {
                let mut fields = self.read_morph_fields(&entry, &line[end + 1..])?;
                if !fields.is_empty() {
                    fields.sort();
                    let joined = fields.join(&u16::from(b' '));
                    morph_id = match morph_indices.get(&joined) {
                        Some(&i) => i,
                        None => {
                            let i = self.d.morph_data.len() as i32;
                            morph_indices.insert(joined.clone(), i);
                            self.d.morph_data.push(joined);
                            i
                        }
                    };
                }
            }
            pending.push((entry, word_form, morph_id));
        }
        let mut builder = WordStorageBuilder::new(word_count, 1.0, has_morph, flags, no_suggest);
        for (entry, form, morph_id) in pending {
            builder.add(&entry, form, morph_id)?;
        }
        self.d.words = builder.finish()?;
        Ok(())
    }

    /// `readMorphFields`: `ph:` fields become `REP` entries, the rest are
    /// returned.
    fn read_morph_fields(
        &mut self,
        word: &[u16],
        unparsed: &[u16],
    ) -> Result<Vec<Vec<u16>>, HunspellError> {
        let mut fields = Vec::new();
        for datum in self.split_morph_data(unparsed)? {
            if datum.starts_with(&u("ph:")) {
                self.add_phonetic_rep_entries(word, &datum[3..]);
            } else {
                fields.push(datum);
            }
        }
        Ok(fields)
    }

    /// `addPhoneticRepEntries`.
    fn add_phonetic_rep_entries(&mut self, word: &[u16], ph: &[u16]) {
        let (mut pattern, mut replacement) = match index_of_str(ph, &u("->"), 0) {
            Some(arrow) if arrow > 0 => (ph[..arrow].to_vec(), ph[arrow + 2..].to_vec()),
            _ => (ph.to_vec(), word.to_vec()),
        };
        if pattern.last() == Some(&u16::from(b'*')) && pattern.len() > 2 && replacement.len() > 1 {
            pattern.truncate(pattern.len() - 2);
            replacement.truncate(replacement.len() - 1);
        }
        if WordCase::case_of(word) == WordCase::Title
            && !pattern.is_empty()
            && WordCase::case_of(&pattern) == WordCase::Lower
        {
            if self.d.has_language(&["de", "hu"]) {
                let lower = self.d.to_lower_case(&replacement);
                self.d.rep_table.push(RepEntry::new(&pattern, &lower));
            }
            let title = self.d.to_title_case(&pattern);
            self.d.rep_table.push(RepEntry::new(&title, &replacement));
        }
        self.d.rep_table.push(RepEntry::new(&pattern, &replacement));
    }
}

/// `StringBuilder.reverse()`: surrogate pairs keep their order.
fn reverse_code_points(s: &[u16]) -> Vec<u16> {
    let mut out: Vec<u16> = s.iter().rev().copied().collect();
    let mut i = 0;
    while i + 1 < out.len() {
        if jc::is_low_surrogate(out[i]) && jc::is_high_surrogate(out[i + 1]) {
            out.swap(i, i + 1);
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

/// `Dictionary.morphBoundary`.
fn morph_boundary(line: &[u16]) -> usize {
    let space_or_tab = |from: usize| -> Option<usize> {
        let tab = index_of(line, u16::from(b'\t'), from);
        let space = index_of(line, u16::from(b' '), from);
        match (tab, space) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    };
    let mut end = match space_or_tab(0) {
        None => return line.len(),
        Some(e) => e,
    };
    loop {
        if end >= line.len() {
            return line.len();
        }
        if line[end] == u16::from(b'\t')
            || end > 0
                && end + 3 < line.len()
                && jc::is_letter(u32::from(line[end + 1]))
                && jc::is_letter(u32::from(line[end + 2]))
                && line[end + 3] == u16::from(b':')
        {
            return end;
        }
        end = match space_or_tab(end + 1) {
            None => return line.len(),
            Some(e) => e,
        };
    }
}

/// `getFlagParsingStrategy`.
fn flag_parsing_strategy(
    flag_line: &[u16],
    charset: Charset,
) -> Result<FlagParsing, HunspellError> {
    let parts = java_split(flag_line, true);
    if parts.len() != 2 {
        return Err(HunspellError::IllegalArgument(format!(
            "Illegal FLAG specification: {}",
            st(flag_line)
        )));
    }
    match st(parts[1]).as_str() {
        "num" => Ok(FlagParsing::Num),
        "UTF-8" => Ok(if charset.java_charset() == Charset::Iso8859_1 {
            FlagParsing::DefaultAsUtf8
        } else {
            FlagParsing::Simple
        }),
        "long" => Ok(FlagParsing::DoubleAscii),
        other => Err(HunspellError::IllegalArgument(format!(
            "Unknown flag type: {other}"
        ))),
    }
}
