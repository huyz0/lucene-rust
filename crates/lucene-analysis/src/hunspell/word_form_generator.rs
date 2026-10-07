//! `WordFormGenerator`: every word form a dictionary entry produces with
//! its affixes ("unmunch").
//!
//! `compress` (`EntrySuggestion`, "munch") is `entry_suggestion.rs`.

use std::collections::BTreeMap;

use super::affix_condition::AffixCondition;
use super::affixed_word::{Affix, AffixedWord};
use super::dictionary::{DictEntry, Dictionary, AFFIX_APPEND, AFFIX_FLAG, AFFIX_STRIP_ORD};
use super::stemmer::Stemmer;
use super::FLAG_UNSET;

/// `WordFormGenerator.AffixEntry`.
#[derive(Debug, Clone)]
struct AffixEntry {
    id: i32,
    prefix: bool,
    affix: Vec<u16>,
    strip: Vec<u16>,
    condition: usize,
}

/// `org.apache.lucene.analysis.hunspell.WordFormGenerator`.
#[derive(Debug)]
pub struct WordFormGenerator<'d> {
    pub(super) dictionary: &'d Dictionary,
    /// Affixes by flag, prefixes first, each in key order.
    affixes: BTreeMap<u16, Vec<AffixEntry>>,
    pub(super) stemmer: Stemmer<'d>,
    /// `NGramFragmentChecker`'s subclass: `canStemToOriginal` always true.
    overgenerate: bool,
}

impl<'d> WordFormGenerator<'d> {
    /// `new WordFormGenerator(dictionary)`.
    pub fn new(dictionary: &'d Dictionary) -> Self {
        let mut g = WordFormGenerator {
            dictionary,
            affixes: BTreeMap::new(),
            stemmer: Stemmer::new(dictionary),
            overgenerate: false,
        };
        for prefix in [true, false] {
            let trie = if prefix {
                &dictionary.prefixes
            } else {
                &dictionary.suffixes
            };
            for (key, ids) in trie.entries() {
                for &id in ids {
                    let flag = dictionary.affix_data(id, AFFIX_FLAG);
                    let affix = if prefix {
                        key.clone()
                    } else {
                        key.iter().rev().copied().collect()
                    };
                    let ord = usize::from(dictionary.affix_data(id, AFFIX_STRIP_ORD));
                    let strip = dictionary.strip_data
                        [dictionary.strip_offsets[ord]..dictionary.strip_offsets[ord + 1]]
                        .to_vec();
                    g.affixes.entry(flag).or_default().push(AffixEntry {
                        id,
                        prefix,
                        affix,
                        strip,
                        condition: dictionary.get_affix_condition(id),
                    });
                }
            }
        }
        g
    }

    /// `getAllWordForms(root)`: the forms of every homonym of `root`.
    pub fn get_all_word_forms(&self, root: &str) -> Vec<AffixedWord> {
        let mut result = Vec::new();
        if let Some(entries) = self.dictionary.lookup_entries(root) {
            for entry in entries {
                result.extend(self.get_all_word_forms_with_flags(root, &entry.flags));
            }
        }
        result
    }

    /// `getAllWordForms(stem, flags)`: `flags` in the dictionary's encoding.
    pub fn get_all_word_forms_with_flags(&self, stem: &str, flags: &str) -> Vec<AffixedWord> {
        let raw: Vec<u16> = flags.encode_utf16().collect();
        let Ok(encoded) = self.dictionary.flag_parsing.parse_utf_flags(&raw) else {
            return vec![];
        };
        self.word_forms_of_flags(stem, flags.to_string(), encoded)
    }

    /// `getAllWordForms(stem, flags)` past the parsing of `flags` (which
    /// `compress` prints from, and would parse back into, `encoded`).
    pub(super) fn word_forms_of_flags(
        &self,
        stem: &str,
        flags: String,
        encoded: Vec<u16>,
    ) -> Vec<AffixedWord> {
        if !self.should_consider_at_all(&encoded) {
            return vec![];
        }
        let entry = DictEntry {
            stem: stem.to_string(),
            flags,
            morphological_data: String::new(),
        };
        self.forms_of(entry, encoded)
    }

    fn forms_of(&self, entry: DictEntry, mut flags: Vec<u16>) -> Vec<AffixedWord> {
        flags.sort_unstable();
        flags.dedup();
        let bare = AffixedWord {
            word: entry.stem.clone(),
            entry,
            prefixes: vec![],
            suffixes: vec![],
        };
        let mut result = Vec::new();
        let d = self.dictionary;
        if !(d.needaffix != FLAG_UNSET && flags.contains(&d.needaffix)) {
            result.push(bare.clone());
        }
        result.extend(self.expand(&bare, &flags));
        result
    }

    /// `generateAllSimpleWords`: every form of every entry, in storage order.
    pub fn generate_all_simple_words(&self, consumer: &mut dyn FnMut(AffixedWord)) {
        let d = self.dictionary;
        let mut entries: Vec<(Vec<u16>, Vec<i32>)> = Vec::new();
        d.words.process_all_words(1, usize::MAX, false, |e| {
            // As Lucene: a second `formStep()` stride over forms already
            // stripped of morphological ids (every other homonym skipped).
            entries.push((e.root.to_vec(), e.forms().step_by(d.form_step()).collect()));
        });
        for (root, forms) in entries {
            let root = String::from_utf16_lossy(&root);
            for form in forms {
                let encoded = d.flag_lookup.get_flags(form);
                if self.should_consider_at_all(&encoded) {
                    let entry = DictEntry {
                        stem: root.clone(),
                        flags: d.flag_parsing.print_flags(&encoded),
                        morphological_data: String::new(),
                    };
                    for aw in self.forms_of(entry, encoded) {
                        consumer(aw);
                    }
                }
            }
        }
    }

    /// `canStemToOriginal`.
    fn can_stem_to_original(&self, derived: &AffixedWord) -> bool {
        if self.overgenerate {
            return true;
        }
        let chars: Vec<u16> = derived.word.encode_utf16().collect();
        if self.is_forbidden_word(&chars) {
            return false;
        }
        let stem: Vec<u16> = derived.entry.stem.encode_utf16().collect();
        let mut found_stem = false;
        let mut found_forbidden = false;
        self.stemmer
            .remove_affixes_for_candidates(&chars, &mut |candidate| {
                if self.is_forbidden_word(candidate) {
                    found_forbidden = true;
                    return false;
                }
                found_stem |= candidate == stem.as_slice();
                !found_stem
            });
        found_stem && !found_forbidden
    }

    fn is_forbidden_word(&self, chars: &[u16]) -> bool {
        let d = self.dictionary;
        d.forbiddenword != FLAG_UNSET
            && !chars.is_empty()
            && d.lookup_word(chars)
                .is_some_and(|forms| d.has_flag_in_forms(forms, d.forbiddenword))
    }

    fn expand(&self, stem: &AffixedWord, flags: &[u16]) -> Vec<AffixedWord> {
        let d = self.dictionary;
        let mut result = Vec::new();
        for &flag in flags {
            let Some(entries) = self.affixes.get(&flag) else {
                continue;
            };
            if !self.is_compatible_with_previous_affixes(stem, entries[0].prefix, flag) {
                continue;
            }
            for affix in entries {
                let Some(derived) = self.apply(affix, stem) else {
                    continue;
                };
                let append_id = d.affix_data(affix.id, AFFIX_APPEND);
                let append = if append_id == 0 {
                    vec![]
                } else {
                    d.flag_lookup.get_flags(i32::from(append_id))
                };
                if self.should_consider_at_all(&append) {
                    if self.can_stem_to_original(&derived) {
                        result.push(derived.clone());
                    }
                    if d.is_cross_product(affix.id) {
                        result.extend(
                            self.expand(&derived, &self.update_flags(flags, flag, &append)),
                        );
                    }
                }
            }
        }
        result
    }

    /// `AffixEntry.apply`.
    fn apply(&self, affix: &AffixEntry, stem: &AffixedWord) -> Option<AffixedWord> {
        let d = self.dictionary;
        let word: Vec<u16> = stem.word.encode_utf16().collect();
        let matches = if affix.prefix {
            word.starts_with(&affix.strip)
        } else {
            word.ends_with(&affix.strip)
        };
        if !matches {
            return None;
        }
        let stripped = if affix.prefix {
            &word[affix.strip.len()..]
        } else {
            &word[..word.len() - affix.strip.len()]
        };
        let accepts = match d.patterns.get(affix.condition) {
            Some(Some(c)) if affix.condition != 0 => c.accepts_stem(stripped),
            _ => AffixCondition::AlwaysTrue.accepts_stem(stripped),
        };
        if !accepts {
            return None;
        }
        let applied = if affix.prefix {
            [affix.affix.as_slice(), stripped].concat()
        } else {
            [stripped, affix.affix.as_slice()].concat()
        };
        let mut prefixes = stem.prefixes.clone();
        let mut suffixes = stem.suffixes.clone();
        let list = if affix.prefix {
            &mut prefixes
        } else {
            &mut suffixes
        };
        list.insert(0, Affix::new(d, affix.id));
        Some(AffixedWord {
            word: String::from_utf16_lossy(&applied),
            entry: stem.entry.clone(),
            prefixes,
            suffixes,
        })
    }

    fn is_compatible_with_previous_affixes(
        &self,
        stem: &AffixedWord,
        is_prefix: bool,
        flag: u16,
    ) -> bool {
        let d = self.dictionary;
        let same = if is_prefix {
            &stem.prefixes
        } else {
            &stem.suffixes
        };
        let size = same.len();
        if size == 2 {
            return false;
        }
        if is_prefix && size == 1 && !d.complex_prefixes {
            return false;
        }
        if !is_prefix && !stem.prefixes.is_empty() {
            return false;
        }
        if size == 1 && !d.is_flag_appended_by_affix(same[0].affix_id, flag) {
            return false;
        }
        true
    }

    fn should_consider_at_all(&self, flags: &[u16]) -> bool {
        let d = self.dictionary;
        !flags.iter().any(|&f| {
            f == d.compound_begin
                || f == d.compound_middle
                || f == d.compound_end
                || f == d.forbiddenword
                || f == d.onlyincompound
        })
    }

    fn update_flags(&self, flags: &[u16], to_remove: u16, to_append: &[u16]) -> Vec<u16> {
        let d = self.dictionary;
        let mut result: Vec<u16> = flags
            .iter()
            .copied()
            .filter(|&f| f != to_remove && f != d.needaffix)
            .collect();
        result.extend_from_slice(to_append);
        result.sort_unstable();
        result.dedup();
        result
    }
}

/// `FragmentChecker`: whether a range of a candidate word overlaps a
/// fragment impossible in the language (lets the suggester skip it).
pub trait FragmentChecker {
    /// `hasImpossibleFragmentAround(word, start, end)` over UTF-16 units.
    fn has_impossible_fragment_around(&self, word: &[u16], start: usize, end: usize) -> bool;
}

/// `FragmentChecker.EVERYTHING_POSSIBLE`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EverythingPossible;

impl FragmentChecker for EverythingPossible {
    fn has_impossible_fragment_around(&self, _word: &[u16], _start: usize, _end: usize) -> bool {
        false
    }
}

/// `NGramFragmentChecker`: the n-grams (n in 2..=4) of the language's words,
/// hashed into a bit set.
#[derive(Debug, Clone)]
pub struct NGramFragmentChecker {
    n: usize,
    hashes: Vec<bool>,
}

/// `NGramFragmentChecker.lowCollisionHash`.
fn low_collision_hash(chars: &[u16]) -> i32 {
    chars
        .iter()
        .fold(0i32, |r, &c| r.wrapping_mul(239).wrapping_add(i32::from(c)))
}

impl NGramFragmentChecker {
    fn checked(n: usize, hashes: Vec<bool>) -> Result<Self, super::HunspellError> {
        if !(2..=4).contains(&n) {
            return Err(super::HunspellError::IllegalArgument(format!(
                "N should be between 2 and 4: {n}"
            )));
        }
        let cardinality = hashes.iter().filter(|&&b| b).count();
        // `BitSet.size()`: the bit capacity, rounded up to whole words.
        if cardinality > hashes.len() * 2 / 3 {
            return Err(super::HunspellError::IllegalArgument(
                "Too many collisions, please report this to dev@lucene.apache.org".into(),
            ));
        }
        Ok(NGramFragmentChecker { n, hashes })
    }

    fn set(hashes: &mut [bool], gram: &[u16]) {
        let len = hashes.len() as i32;
        hashes[(low_collision_hash(gram) % len).unsigned_abs() as usize] = true;
    }

    /// `java.util.BitSet(nbits)`'s `size()`: `nbits` rounded up to 64.
    fn bitset(nbits: usize) -> Vec<bool> {
        vec![false; nbits.div_ceil(64).max(1) * 64]
    }

    /// `fromWords(n, words)`.
    pub fn from_words(n: usize, words: &[&str]) -> Result<Self, super::HunspellError> {
        let highest = if words.is_empty() {
            0
        } else {
            1usize << (usize::BITS - 1 - words.len().leading_zeros())
        };
        let mut hashes = Self::bitset(highest * 4);
        for w in words {
            let units: Vec<u16> = w.encode_utf16().collect();
            if units.len() >= n {
                for i in 0..=units.len() - n {
                    Self::set(&mut hashes, &units[i..i + n]);
                }
            }
        }
        Self::checked(n, hashes)
    }

    /// `fromAllSimpleWords(n, dictionary)`: the n-grams of every generated
    /// word form (overgenerating, as Lucene skips `canStemToOriginal`), with
    /// upper-case and capitalized variants of cased entries.
    pub fn from_all_simple_words(
        n: usize,
        dictionary: &Dictionary,
    ) -> Result<Self, super::HunspellError> {
        let mut hashes = Self::bitset(1 << (7 + n * 3));
        let mut gen = WordFormGenerator::new(dictionary);
        gen.overgenerate = true;
        let grams = |word: &[u16], hashes: &mut Vec<bool>| {
            if word.len() >= n {
                for i in 0..=word.len() - n {
                    Self::set(hashes, &word[i..i + n]);
                }
            }
        };
        gen.generate_all_simple_words(&mut |aw| {
            let word: Vec<u16> = aw.word.encode_utf16().collect();
            grams(&word, &mut hashes);
            let stem: Vec<u16> = aw.entry.stem.encode_utf16().collect();
            let case = super::word_case::WordCase::case_of(&stem);
            if case != super::word_case::WordCase::Mixed
                && case != super::word_case::WordCase::Neutral
            {
                grams(&super::suggester::java_upper(&word), &mut hashes);
                if word.len() > 1 {
                    let mut cap = vec![super::word_case::to_upper(word[0])];
                    cap.extend_from_slice(&word[1..n.min(word.len())]);
                    grams(&cap, &mut hashes);
                }
            }
        });
        Self::checked(n, hashes)
    }
}

impl FragmentChecker for NGramFragmentChecker {
    fn has_impossible_fragment_around(&self, word: &[u16], start: usize, end: usize) -> bool {
        if word.len() < self.n {
            return false;
        }
        let first = start.saturating_sub(self.n - 1);
        let last = (end as isize - 1).min((word.len() - self.n) as isize);
        let len = self.hashes.len() as i32;
        let mut i = first as isize;
        while i <= last {
            let gram = &word[i as usize..i as usize + self.n];
            if !self.hashes[(low_collision_hash(gram) % len).unsigned_abs() as usize] {
                return true;
            }
            i += 1;
        }
        false
    }
}
