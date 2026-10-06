//! `org.apache.lucene.analysis.pt`: `PortugueseAnalyzer`, the light, minimal
//! and RSLP stemmers, and `RSLPStemmerBase` (shared with Galician).
//!
//! RSLP rules are Lucene's own `portuguese.rslp`/`galician.rslp`, vendored,
//! parsed once by [`parse_rslp`] (Java's four rule patterns, matched with the
//! `regex` crate: the files are fixed ASCII-structured resources).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::{ends_with, ends_with_units};
use crate::CharArraySet;

use super::{mark_exclusions, snowball_set, std_lower_stop, CharStemmer, StemFilter};

const fn c(ch: char) -> u16 {
    ch as u16
}

/// `PortugueseAnalyzer.getDefaultStopSet()` (`snowball/portuguese_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/portuguese_stop.txt")));

/// `RSLPStemmerBase.Rule` and its two exception subclasses.
#[derive(Debug)]
pub(crate) struct Rule {
    suffix: Vec<u16>,
    replacement: Vec<u16>,
    min: usize,
    /// `RuleWithSetExceptions`: whole words exempt.
    set_exceptions: Option<HashSet<Vec<u16>>>,
    /// `RuleWithSuffixExceptions`: endings exempt.
    suffix_exceptions: Vec<Vec<u16>>,
}

impl Rule {
    // Java: Rule.matches (+ the subclasses' overrides)
    fn matches(&self, s: &[u16], len: usize) -> bool {
        if !(len >= self.min + self.suffix.len() && ends_with_units(s, len, &self.suffix)) {
            return false;
        }
        if let Some(set) = &self.set_exceptions {
            return !set.contains(&s[..len]);
        }
        !self
            .suffix_exceptions
            .iter()
            .any(|e| ends_with_units(s, len, e))
    }

    // Java: Rule.replace
    fn replace(&self, s: &mut Vec<u16>, len: usize) -> usize {
        s.truncate(len - self.suffix.len());
        s.extend_from_slice(&self.replacement);
        s.len()
    }
}

/// `RSLPStemmerBase.Step`.
#[derive(Debug)]
pub(crate) struct Step {
    rules: Vec<Rule>,
    min: usize,
    suffixes: Vec<Vec<u16>>,
}

impl Step {
    /// `apply(char[], int)`.
    pub(crate) fn apply(&self, s: &mut Vec<u16>, len: usize) -> usize {
        if len < self.min {
            return len;
        }
        if !self.suffixes.is_empty() && !self.suffixes.iter().any(|x| ends_with_units(s, len, x)) {
            return len;
        }
        for r in &self.rules {
            if r.matches(s, len) {
                return r.replace(s, len);
            }
        }
        len
    }
}

fn units(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// `RSLPStemmerBase.parseList`.
fn parse_list(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|x| {
            let t = x.trim();
            t[1..t.len() - 1].to_string()
        })
        .collect()
}

/// `RSLPStemmerBase.parse`: the steps of a rules file, by name. Panics on a
/// malformed file (Java throws `RuntimeException`; the files are vendored).
pub(crate) fn parse_rslp(text: &str) -> HashMap<String, Step> {
    let header =
        regex::Regex::new(r#"^\{\s*"([^"]*)",\s*([0-9]+),\s*(0|1),\s*\{(.*)\},\s*$"#).unwrap();
    let strip = regex::Regex::new(r#"^\{\s*"([^"]*)",\s*([0-9]+)\s*\}\s*(,|(\}\s*;))$"#).unwrap();
    let rep =
        regex::Regex::new(r#"^\{\s*"([^"]*)",\s*([0-9]+),\s*"([^"]*)"\}\s*(,|(\}\s*;))$"#).unwrap();
    let exc = regex::Regex::new(
        r#"^\{\s*"([^"]*)",\s*([0-9]+),\s*"([^"]*)",\s*\{(.*)\}\s*\}\s*(,|(\}\s*;))$"#,
    )
    .unwrap();
    // Java: readLine (trimmed, skipping blanks and '#' comments)
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let mut steps = HashMap::new();
    while let Some(h) = lines.next() {
        let m = header.captures(h).expect("an RSLP step header");
        let name = m[1].to_string();
        let mut min: usize = m[2].parse().unwrap();
        let ty: u8 = m[3].parse().unwrap();
        let suffixes = parse_list(&m[4]);
        let mut rules = Vec::new();
        for line in lines.by_ref() {
            let rule = if let Some(m) = strip.captures(line) {
                (
                    m[1].to_string(),
                    m[2].parse().unwrap(),
                    String::new(),
                    Vec::new(),
                )
            } else if let Some(m) = rep.captures(line) {
                (
                    m[1].to_string(),
                    m[2].parse().unwrap(),
                    m[3].to_string(),
                    Vec::new(),
                )
            } else {
                let m = exc.captures(line).expect("an RSLP rule");
                (
                    m[1].to_string(),
                    m[2].parse().unwrap(),
                    m[3].to_string(),
                    parse_list(&m[4]),
                )
            };
            let (suffix, rmin, replacement, exceptions) = rule;
            for e in &exceptions {
                assert!(e.ends_with(&suffix), "useless exception '{e}'");
            }
            let (set_exceptions, suffix_exceptions) = if exceptions.is_empty() {
                (None, Vec::new())
            } else if ty == 0 {
                (None, exceptions.iter().map(|e| units(e)).collect())
            } else {
                (
                    Some(exceptions.iter().map(|e| units(e)).collect()),
                    Vec::new(),
                )
            };
            rules.push(Rule {
                suffix: units(&suffix),
                replacement: units(&replacement),
                min: rmin,
                set_exceptions,
                suffix_exceptions,
            });
            if line.ends_with(';') {
                break;
            }
        }
        if min == 0 {
            min = rules
                .iter()
                .map(|r| r.min + r.suffix.len())
                .min()
                .unwrap_or(usize::MAX);
        }
        steps.insert(
            name,
            Step {
                rules,
                min,
                suffixes: suffixes.iter().map(|x| units(x)).collect(),
            },
        );
    }
    steps
}

static PT_STEPS: LazyLock<HashMap<String, Step>> =
    LazyLock::new(|| parse_rslp(include_str!("stopwords/portuguese.rslp")));

fn step(name: &str) -> &'static Step {
    &PT_STEPS[name]
}

/// `PortugueseStemmer` (RSLP, Orengo & Huyck).
#[derive(Debug, Default, Clone, Copy)]
pub struct PortugueseStemmer;

impl CharStemmer for PortugueseStemmer {
    // Java: PortugueseStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        for name in ["Plural", "Adverb", "Feminine", "Augmentative"] {
            len = step(name).apply(s, len);
        }
        let old = len;
        len = step("Noun").apply(s, len);
        if len == old {
            let old = len;
            len = step("Verb").apply(s, len);
            if len == old {
                len = step("Vowel").apply(s, len);
            }
        }
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0xE0..=0xE5 => c('a'),
                0xE7 => c('c'),
                0xE8..=0xEB => c('e'),
                0xEC..=0xEF => c('i'),
                0xF1 => c('n'),
                0xF2..=0xF6 => c('o'),
                0xF9..=0xFC => c('u'),
                0xFD | 0xFF => c('y'),
                o => o,
            };
        }
        len
    }
}

/// `PortugueseMinimalStemmer`: RSLP's plural step.
#[derive(Debug, Default, Clone, Copy)]
pub struct PortugueseMinimalStemmer;

impl CharStemmer for PortugueseMinimalStemmer {
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        step("Plural").apply(s, len)
    }
}

/// `PortugueseLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PortugueseLightStemmer;

impl PortugueseLightStemmer {
    // Java: PortugueseLightStemmer.removeSuffix
    fn remove_suffix(s: &mut [u16], len: usize) -> usize {
        if len > 4 && ends_with(s, len, "es") && matches!(s[len - 3], 0x72 | 0x73 | 0x6C | 0x7A) {
            return len - 2;
        }
        if len > 3 && ends_with(s, len, "ns") {
            s[len - 2] = c('m');
            return len - 1;
        }
        if len > 4 && (ends_with(s, len, "eis") || ends_with(s, len, "éis")) {
            s[len - 3] = c('e');
            s[len - 2] = c('l');
            return len - 1;
        }
        if len > 4 && ends_with(s, len, "ais") {
            s[len - 2] = c('l');
            return len - 1;
        }
        if len > 4 && ends_with(s, len, "óis") {
            s[len - 3] = c('o');
            s[len - 2] = c('l');
            return len - 1;
        }
        if len > 4 && ends_with(s, len, "is") {
            s[len - 1] = c('l');
            return len;
        }
        if len > 3 && (ends_with(s, len, "ões") || ends_with(s, len, "ães")) {
            let len = len - 1;
            s[len - 2] = c('ã');
            s[len - 1] = c('o');
            return len;
        }
        if len > 6 && ends_with(s, len, "mente") {
            return len - 5;
        }
        if len > 3 && s[len - 1] == c('s') {
            return len - 1;
        }
        len
    }

    // Java: PortugueseLightStemmer.normFeminine
    fn norm_feminine(s: &mut [u16], len: usize) -> usize {
        let e = |s: &[u16], x: &str| ends_with(s, len, x);
        if len > 7 && (e(s, "inha") || e(s, "iaca") || e(s, "eira")) {
            s[len - 1] = c('o');
            return len;
        }
        if len > 6 {
            if ["osa", "ica", "ida", "ada", "iva", "ama"]
                .iter()
                .any(|x| e(s, x))
            {
                s[len - 1] = c('o');
                return len;
            }
            if e(s, "ona") {
                s[len - 3] = c('ã');
                s[len - 2] = c('o');
                return len - 1;
            }
            if e(s, "ora") {
                return len - 1;
            }
            if e(s, "esa") {
                s[len - 3] = c('ê');
                return len - 1;
            }
            if e(s, "na") {
                s[len - 1] = c('o');
                return len;
            }
        }
        len
    }
}

impl CharStemmer for PortugueseLightStemmer {
    // Java: PortugueseLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        if len < 4 {
            return len;
        }
        len = Self::remove_suffix(s, len);
        if len > 3 && s[len - 1] == c('a') {
            len = Self::norm_feminine(s, len);
        }
        if len > 4 && matches!(s[len - 1], 0x65 | 0x61 | 0x6F) {
            len -= 1;
        }
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0xE0 | 0xE1 | 0xE2 | 0xE4 | 0xE3 => c('a'),
                0xF2 | 0xF3 | 0xF4 | 0xF6 | 0xF5 => c('o'),
                0xE8 | 0xE9 | 0xEA | 0xEB => c('e'),
                0xF9 | 0xFA | 0xFB | 0xFC => c('u'),
                0xEC | 0xED | 0xEE | 0xEF => c('i'),
                0xE7 => c('c'),
                o => o,
            };
        }
        len
    }
}

/// `PortugueseLightStemFilter`.
pub type PortugueseLightStemFilter<I> = StemFilter<I, PortugueseLightStemmer>;
/// `PortugueseMinimalStemFilter`.
pub type PortugueseMinimalStemFilter<I> = StemFilter<I, PortugueseMinimalStemmer>;
/// `PortugueseStemFilter`.
pub type PortugueseStemFilter<I> = StemFilter<I, PortugueseStemmer>;

language_analyzer! {
    /// `PortugueseAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, [`PortugueseLightStemFilter`].
    PortugueseAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        PortugueseLightStemFilter::new(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
