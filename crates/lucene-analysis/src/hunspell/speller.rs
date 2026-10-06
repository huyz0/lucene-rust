//! `org.apache.lucene.analysis.hunspell.Hunspell`: spell checking, roots and
//! analyses over a [`Dictionary`].

use super::affixed_word::{Affix, AffixedWord};
use super::dictionary::Dictionary;
use super::stemmer::Stemmer;
use super::timeout::{Canceler, CheckCanceled, TimeoutPolicy};

/// `org.apache.lucene.analysis.hunspell.Hunspell`: spell checking,
/// suggestions, roots and analyses, under a [`TimeoutPolicy`] and an
/// optional [`CheckCanceled`] hook.
#[derive(Clone, Copy)]
pub struct Hunspell<'d> {
    pub(crate) dictionary: &'d Dictionary,
    pub(crate) stemmer: Stemmer<'d>,
    /// The `Suggester`'s speller: `acceptsStem` refuses `NOSUGGEST` and
    /// `SUBSTANDARD` roots.
    pub(crate) suggestion_mode: bool,
    /// What `suggest` does when it runs out of time.
    pub(crate) policy: TimeoutPolicy,
    /// The caller's `checkCanceled`.
    pub(crate) check_canceled: Option<CheckCanceled<'d>>,
    /// The running computation's `checkCanceled` (`None`: never stops).
    pub(crate) cancel: Option<&'d Canceler<'d>>,
}

impl std::fmt::Debug for Hunspell<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hunspell")
            .field("suggestion_mode", &self.suggestion_mode)
            .field("policy", &self.policy)
            .field("check_canceled", &self.check_canceled.is_some())
            .field("cancel", &self.cancel)
            .finish_non_exhaustive()
    }
}

impl<'d> Hunspell<'d> {
    /// `new Hunspell(dictionary)`: `TimeoutPolicy.RETURN_PARTIAL_RESULT`,
    /// no cancellation hook.
    pub fn new(dictionary: &'d Dictionary) -> Self {
        Self::with_timeout_policy(dictionary, TimeoutPolicy::ReturnPartialResult, None)
    }

    /// `new Hunspell(dictionary, policy, checkCanceled)`.
    pub fn with_timeout_policy(
        dictionary: &'d Dictionary,
        policy: TimeoutPolicy,
        check_canceled: Option<CheckCanceled<'d>>,
    ) -> Self {
        Hunspell {
            dictionary,
            stemmer: Stemmer::new(dictionary),
            suggestion_mode: false,
            policy,
            check_canceled,
            cancel: None,
        }
    }

    /// `checkCanceled.run()`: whether the running computation must stop.
    #[inline]
    pub(crate) fn canceled(&self) -> bool {
        self.cancel.is_some_and(Canceler::check)
    }

    /// `getRoots(word)`: the distinct stems.
    pub fn get_roots(&self, word: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for s in self.stemmer.stem(word) {
            if !out.contains(&s) {
                out.push(s);
            }
        }
        out
    }

    /// `analyzeSimpleWord(word)`.
    pub fn analyze_simple_word(&self, word: &str) -> Vec<AffixedWord> {
        let d = self.dictionary;
        let units: Vec<u16> = word.encode_utf16().collect();
        let mut result = Vec::new();
        self.stemmer.analyze(&units, &mut |m| {
            let affixes = |ids: [i32; 2]| -> Vec<Affix> {
                ids.iter()
                    .filter(|&&id| id >= 0)
                    .map(|&id| Affix::new(d, id))
                    .collect()
            };
            let prefixes = affixes([m.outer_prefix, m.inner_prefix]);
            let suffixes = affixes([m.outer_suffix, m.inner_suffix]);
            result.push(AffixedWord {
                word: word.to_string(),
                entry: d.dict_entry(m.stem, m.form_id, m.morph_data_id),
                prefixes,
                suffixes,
            });
            true
        });
        result
    }
}

/// `Root<CharsRef>`: a dictionary root found for a word part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Root {
    pub(crate) word: Vec<u16>,
    pub(crate) entry_id: i32,
}

/// A `CharsRef`: `length` units at `offset` of a buffer whose start
/// `hasForceUCaseProblem` inspects.
#[derive(Debug, Clone, Copy)]
struct CharsRef<'a> {
    chars: &'a [u16],
    offset: usize,
    length: usize,
}

impl<'a> CharsRef<'a> {
    fn units(&self) -> &'a [u16] {
        &self.chars[self.offset..self.offset + self.length]
    }
}

/// `Hunspell.CompoundPart`.
struct CompoundPart<'a> {
    index: i32,
    length: usize,
    tail: CharsRef<'a>,
    root: Root,
    enabling_pattern: Option<&'a CheckCompoundPattern>,
}

use super::dictionary::CheckCompoundPattern;
use super::stemmer::WordContext;
use super::word_case::{is_upper, WordCase};
use super::FLAG_UNSET;
use super::HIDDEN_FLAG;

impl<'d> Hunspell<'d> {
    /// `spell(word)`: whether `word` is correctly spelled (`false` once the
    /// [`CheckCanceled`] hook cancels it).
    pub fn spell(&self, word: &str) -> bool {
        let units: Vec<u16> = word.encode_utf16().collect();
        match (self.cancel, self.check_canceled) {
            (None, Some(hook)) => {
                let cancel = Canceler::new(Some(hook), None);
                Hunspell {
                    cancel: Some(&cancel),
                    ..*self
                }
                .spell_units(&units)
            }
            _ => self.spell_units(&units),
        }
    }

    pub(crate) fn spell_units(&self, word: &[u16]) -> bool {
        if self.canceled() {
            return false;
        }
        if word.is_empty() {
            return true;
        }
        let cleaned;
        let mut word = word;
        if self.dictionary.needs_input_cleaning(word) {
            cleaned = self.dictionary.clean_input(word);
            word = &cleaned;
        }
        if word.last() == Some(&u16::from(b'.')) {
            return self.spell_with_trailing_dots(word);
        }
        self.spell_clean(word)
    }

    fn spell_clean(&self, word: &[u16]) -> bool {
        if is_number(word) {
            return true;
        }
        if let Some(simple) = self.check_simple_word(word, None) {
            return simple;
        }
        if self.check_compounds_word(word, None) {
            return true;
        }
        let wc = self.stemmer.case_of(word);
        if (wc == WordCase::Upper || wc == WordCase::Title)
            && !self.stemmer.vary_case(word, wc, &mut |variant, original| {
                !self.check_word(variant, original)
            })
        {
            return true;
        }
        if self.dictionary.breaks.is_not_empty() && !self.has_too_many_break_occurrences(word) {
            return self.try_breaks(word);
        }
        false
    }

    fn spell_with_trailing_dots(&self, word: &[u16]) -> bool {
        let mut length = word.len() - 1;
        while length > 0 && word[length - 1] == u16::from(b'.') {
            length -= 1;
        }
        self.spell_clean(&word[..length]) || self.spell_clean(&word[..length + 1])
    }

    /// `checkSimpleWord`: `Some(not forbidden)` when a root is found.
    pub(crate) fn check_simple_word(
        &self,
        word: &[u16],
        original: Option<WordCase>,
    ) -> Option<bool> {
        let entry = self.find_stem(word, original, WordContext::SimpleWord)?;
        Some(
            !self
                .dictionary
                .has_flag(entry.entry_id, self.dictionary.forbiddenword),
        )
    }

    /// `checkWord(char[], int, WordCase)`.
    pub(crate) fn check_word(&self, word: &[u16], original: Option<WordCase>) -> bool {
        if let Some(simple) = self.check_simple_word(word, original) {
            return simple;
        }
        self.check_compounds_word(word, original)
    }

    /// `checkCompounds(char[], int, WordCase)`.
    fn check_compounds_word(&self, word: &[u16], original: Option<WordCase>) -> bool {
        let d = self.dictionary;
        if d.compound_rules.is_some() && self.check_compound_rules(word, &mut Vec::new()) {
            return true;
        }
        if d.compound_begin != FLAG_UNSET || d.compound_flag != FLAG_UNSET {
            let whole = CharsRef {
                chars: word,
                offset: 0,
                length: word.len(),
            };
            return self.check_compounds(whole, original, None);
        }
        false
    }

    /// `findStem`: the first acceptable root of `word` in `context`.
    pub(crate) fn find_stem(
        &self,
        word: &[u16],
        original: Option<WordCase>,
        context: WordContext,
    ) -> Option<Root> {
        if self.canceled() {
            return None;
        }
        let d = self.dictionary;
        let to_check =
            if context != WordContext::CompoundMiddle && context != WordContext::CompoundEnd {
                original
            } else {
                None
            };
        let mut result = None;
        self.stemmer.do_stem(word, context, &mut |m| {
            if !self.accept_case(to_check, m.form_id, m.stem) {
                return d.has_flag(m.form_id, HIDDEN_FLAG);
            }
            if self.accepts_stem(m.form_id) {
                result = Some(Root {
                    word: m.stem.to_vec(),
                    entry_id: m.form_id,
                });
            }
            false
        });
        result
    }

    /// `acceptsStem`: every root, or for the suggestion speller every root
    /// without `NOSUGGEST`/`SUBSTANDARD`.
    fn accepts_stem(&self, form_id: i32) -> bool {
        let d = self.dictionary;
        !self.suggestion_mode
            || !d.has_flag(form_id, d.no_suggest) && !d.has_flag(form_id, d.sub_standard)
    }

    /// `acceptCase`.
    fn accept_case(&self, original: Option<WordCase>, entry_id: i32, root: &[u16]) -> bool {
        let d = self.dictionary;
        let keep_case = d.has_flag(entry_id, d.keepcase);
        if let Some(original) = original {
            if keep_case && d.check_sharp_s && original == WordCase::Title && root.contains(&0x00DF)
            {
                return true;
            }
            return !keep_case;
        }
        !d.has_flag(entry_id, HIDDEN_FLAG)
    }

    /// `checkCompounds(CharsRef, WordCase, CompoundPart)`.
    fn check_compounds(
        &self,
        word: CharsRef<'_>,
        original: Option<WordCase>,
        prev: Option<&CompoundPart<'_>>,
    ) -> bool {
        let d = self.dictionary;
        if let Some(p) = prev {
            if p.index > d.compound_max - 2 {
                return false;
            }
        }
        let min = d.compound_min as usize;
        let limit = (word.length + 1).saturating_sub(min);
        let mut break_pos = min;
        while break_pos < limit {
            let context = if prev.is_none() {
                WordContext::CompoundBegin
            } else {
                WordContext::CompoundMiddle
            };
            let break_offset = word.offset + break_pos;
            if self.may_break_into_compounds(word.chars, word.offset, word.length, break_offset) {
                let mut stem = self.find_stem(
                    &word.chars[word.offset..word.offset + break_pos],
                    original,
                    context,
                );
                if stem.is_none()
                    && d.simplified_triple
                    && word.chars[break_offset - 1] == word.chars[break_offset]
                {
                    stem = self.find_stem(
                        &word.chars[word.offset..word.offset + break_pos + 1],
                        original,
                        context,
                    );
                }
                if let Some(stem) = stem {
                    if !d.has_flag(stem.entry_id, d.forbiddenword)
                        && prev.is_none_or(|p| self.may_compound(p, &stem, break_pos, original))
                    {
                        let part = CompoundPart {
                            index: prev.map_or(1, |p| p.index + 1),
                            length: break_pos,
                            tail: word,
                            root: stem,
                            enabling_pattern: None,
                        };
                        if self.check_compounds_after(original, &part) {
                            return true;
                        }
                    }
                }
            }
            if self.check_compound_pattern_replacements(word, break_pos, original, prev) {
                return true;
            }
            break_pos += 1;
        }
        false
    }

    fn check_compound_pattern_replacements(
        &self,
        word: CharsRef<'_>,
        pos: usize,
        original: Option<WordCase>,
        prev: Option<&CompoundPart<'_>>,
    ) -> bool {
        let d = self.dictionary;
        for pattern in &d.check_compound_patterns {
            let Some(expanded) =
                pattern.expand_replacement(word.chars, word.offset, word.length, pos)
            else {
                continue;
            };
            let context = if prev.is_none() {
                WordContext::CompoundBegin
            } else {
                WordContext::CompoundMiddle
            };
            let break_pos = pos + pattern.end_length();
            if let Some(stem) = self.find_stem(
                &expanded[..break_pos.min(expanded.len())],
                original,
                context,
            ) {
                let part = CompoundPart {
                    index: prev.map_or(1, |p| p.index + 1),
                    length: break_pos,
                    tail: CharsRef {
                        chars: &expanded,
                        offset: 0,
                        length: expanded.len(),
                    },
                    root: stem,
                    enabling_pattern: Some(pattern),
                };
                if self.check_compounds_after(original, &part) {
                    return true;
                }
            }
        }
        false
    }

    fn check_compounds_after(&self, original: Option<WordCase>, prev: &CompoundPart<'_>) -> bool {
        let d = self.dictionary;
        let word = prev.tail;
        let break_pos = prev.length;
        let remaining = word.length - break_pos;
        let break_offset = word.offset + break_pos;
        let last_root = self.find_stem(
            &word.chars[break_offset..break_offset + remaining],
            original,
            WordContext::CompoundEnd,
        );
        if let Some(last) = &last_root {
            if !d.has_flag(last.entry_id, d.forbiddenword)
                && !(d.check_compound_dup && prev.root == *last)
                && !self.has_force_u_case_problem(last, original, word.chars)
                && self.may_compound(prev, last, remaining, original)
            {
                return true;
            }
        }
        let tail = CharsRef {
            chars: word.chars,
            offset: break_offset,
            length: remaining,
        };
        self.check_compounds(tail, original, Some(prev))
    }

    fn has_force_u_case_problem(
        &self,
        root: &Root,
        original: Option<WordCase>,
        chars: &[u16],
    ) -> bool {
        if original == Some(WordCase::Title) || original == Some(WordCase::Upper) {
            return false;
        }
        if original.is_none() && chars.first().is_some_and(|&c| is_upper(c)) {
            return false;
        }
        self.dictionary
            .has_flag(root.entry_id, self.dictionary.force_u_case)
    }

    /// `CompoundPart.mayCompound`.
    fn may_compound(
        &self,
        part: &CompoundPart<'_>,
        next: &Root,
        next_len: usize,
        original: Option<WordCase>,
    ) -> bool {
        let d = self.dictionary;
        let tail = part.tail.units();
        let before = (part.root.word.as_slice(), part.root.entry_id);
        let after = (next.word.as_slice(), next.entry_id);
        let patterns_ok = match part.enabling_pattern {
            Some(p) => p.prohibits_compounding(d, tail, part.length, before, after),
            None => !d
                .check_compound_patterns
                .iter()
                .any(|p| p.prohibits_compounding(d, tail, part.length, before, after)),
        };
        if !patterns_ok {
            return false;
        }
        if d.check_compound_rep
            && self.is_misspelled_simple_word(&tail[..part.length + next_len], original)
        {
            return false;
        }
        let mut spaced = Vec::with_capacity(part.length + next_len + 1);
        spaced.extend_from_slice(&tail[..part.length]);
        spaced.push(u16::from(b' '));
        spaced.extend_from_slice(&tail[part.length..part.length + next_len]);
        self.check_simple_word(&spaced, None) != Some(true)
    }

    /// `CompoundPart.isMisspelledSimpleWord`.
    fn is_misspelled_simple_word(&self, word: &[u16], original: Option<WordCase>) -> bool {
        for entry in &self.dictionary.rep_table {
            if entry.is_middle() {
                for sug in entry.substitute(word) {
                    if self
                        .find_stem(&sug, original, WordContext::SimpleWord)
                        .is_some()
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// `mayBreakIntoCompounds`.
    fn may_break_into_compounds(
        &self,
        chars: &[u16],
        offset: usize,
        length: usize,
        break_pos: usize,
    ) -> bool {
        let d = self.dictionary;
        let dash = u16::from(b'-');
        if d.check_compound_case {
            let (a, b) = (chars[break_pos - 1], chars[break_pos]);
            if (is_upper(a) || is_upper(b)) && a != dash && b != dash {
                return false;
            }
        }
        if d.check_compound_triple
            && chars[break_pos - 1] == chars[break_pos]
            && (break_pos > offset + 1 && chars[break_pos - 2] == chars[break_pos - 1]
                || break_pos + 1 < length && chars[break_pos] == chars[break_pos + 1])
        {
            return false;
        }
        true
    }

    /// `checkCompoundRules`.
    fn check_compound_rules(&self, word: &[u16], words: &mut Vec<Vec<i32>>) -> bool {
        let d = self.dictionary;
        if words.len() >= 100 || self.canceled() {
            return false;
        }
        let min = d.compound_min as usize;
        let limit = (word.len() + 1).saturating_sub(min);
        for break_pos in min..limit {
            let Some(forms) = d.lookup_word(&word[..break_pos]) else {
                continue;
            };
            words.push(forms.to_vec());
            if self.may_have_compound_rule(words) {
                if self.check_last_compound_part(&word[break_pos..], words) {
                    return true;
                }
                if self.check_compound_rules(&word[break_pos..], words) {
                    return true;
                }
            }
            words.pop();
        }
        false
    }

    fn may_have_compound_rule(&self, words: &[Vec<i32>]) -> bool {
        self.dictionary
            .compound_rules
            .iter()
            .flatten()
            .any(|r| r.may_match(self.dictionary, words))
    }

    /// `checkLastCompoundPart`.
    fn check_last_compound_part(&self, word: &[u16], words: &mut Vec<Vec<i32>>) -> bool {
        let d = self.dictionary;
        words.push(vec![0]);
        let found = !self
            .stemmer
            .do_stem(word, WordContext::CompoundRuleEnd, &mut |m| {
                if let Some(last) = words.last_mut() {
                    last[0] = m.form_id;
                }
                !d.compound_rules
                    .iter()
                    .flatten()
                    .any(|r| r.fully_matches(d, words))
            });
        words.pop();
        found
    }

    /// `tryBreaks`.
    fn try_breaks(&self, word: &[u16]) -> bool {
        let breaks = &self.dictionary.breaks;
        for br in &breaks.starting {
            if word.len() > br.len() && word.starts_with(br) && self.spell_units(&word[br.len()..])
            {
                return true;
            }
        }
        for br in &breaks.ending {
            if word.len() > br.len()
                && word.ends_with(br)
                && self.spell_units(&word[..word.len() - br.len()])
            {
                return true;
            }
        }
        for br in &breaks.middle {
            let pos = super::dictionary::index_of_str(word, br, 0);
            if self.can_be_broken_at(word, br, pos) {
                return true;
            }
            if let Some(p) = pos {
                if p > 0
                    && self.can_be_broken_at(
                        word,
                        br,
                        super::dictionary::index_of_str(word, br, p + 1),
                    )
                {
                    return true;
                }
            }
        }
        false
    }

    fn has_too_many_break_occurrences(&self, word: &[u16]) -> bool {
        let mut occurrences = 0;
        for br in &self.dictionary.breaks.middle {
            let mut pos = 0;
            while let Some(p) = super::dictionary::index_of_str(word, br, pos) {
                occurrences += 1;
                if occurrences >= 10 {
                    return true;
                }
                pos = p + br.len();
            }
        }
        false
    }

    fn can_be_broken_at(&self, word: &[u16], br: &[u16], pos: Option<usize>) -> bool {
        let Some(pos) = pos else {
            return false;
        };
        pos > 0
            && pos + br.len() < word.len()
            && self.spell_units(&word[..pos])
            && self.spell_units(&word[pos + br.len()..])
    }
}

/// `Hunspell.isNumber`.
fn is_number(s: &[u16]) -> bool {
    let digit = |c: u16| (u16::from(b'0')..=u16::from(b'9')).contains(&c);
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        if digit(c) {
            i += 1;
        } else if c == u16::from(b'.') || c == u16::from(b',') || c == u16::from(b'-') {
            if i == 0 || i + 1 >= s.len() || !digit(s[i + 1]) {
                return false;
            }
            i += 2;
        } else {
            return false;
        }
    }
    true
}
