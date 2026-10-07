//! `org.apache.lucene.analysis.hunspell.Stemmer`: finds the dictionary roots
//! of a word by removing up to two suffixes and two prefixes (or the
//! reverse under `COMPLEXPREFIXES`), honouring every affix and root flag.

use super::dictionary::{Dictionary, AFFIX_APPEND, AFFIX_FLAG, AFFIX_STRIP_ORD};
use super::word_case::{to_upper, WordCase};
use super::FLAG_UNSET;

/// `WordContext`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WordContext {
    /// `SIMPLE_WORD`.
    SimpleWord,
    /// `COMPOUND_BEGIN`.
    CompoundBegin,
    /// `COMPOUND_MIDDLE`.
    CompoundMiddle,
    /// `COMPOUND_END`.
    CompoundEnd,
    /// `COMPOUND_RULE_END`.
    CompoundRuleEnd,
}

impl WordContext {
    /// `isCompound`.
    pub(crate) fn is_compound(self) -> bool {
        self != WordContext::SimpleWord
    }

    /// `isAffixAllowedWithoutSpecialPermit`.
    fn is_affix_allowed_without_special_permit(self, is_prefix: bool) -> bool {
        if is_prefix {
            self == WordContext::CompoundBegin
        } else {
            self == WordContext::CompoundEnd || self == WordContext::CompoundRuleEnd
        }
    }

    /// `requiredFlag`.
    fn required_flag(self, d: &Dictionary) -> u16 {
        match self {
            WordContext::CompoundBegin => d.compound_begin,
            WordContext::CompoundMiddle => d.compound_middle,
            WordContext::CompoundEnd => d.compound_end,
            WordContext::CompoundRuleEnd | WordContext::SimpleWord => FLAG_UNSET,
        }
    }
}

/// A root found for a word (`RootProcessor.processRoot`'s arguments).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RootMatch<'a> {
    /// The root's text.
    pub(crate) stem: &'a [u16],
    /// `formID`: the root's flag-set id.
    pub(crate) form_id: i32,
    /// `morphDataId`.
    pub(crate) morph_data_id: i32,
    pub(crate) outer_prefix: i32,
    pub(crate) inner_prefix: i32,
    pub(crate) outer_suffix: i32,
    pub(crate) inner_suffix: i32,
}

/// A stripped word to look up (`StemCandidateProcessor.processStemCandidate`'s
/// arguments).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate<'a> {
    pub(crate) word: &'a [u16],
    pub(crate) last_affix: i32,
    pub(crate) outer_prefix: i32,
    pub(crate) inner_prefix: i32,
    pub(crate) outer_suffix: i32,
    pub(crate) inner_suffix: i32,
}

/// The affixes removed so far (Java's `outerPrefix`, `innerPrefix`,
/// `outerSuffix` arguments).
#[derive(Debug, Clone, Copy)]
struct Applied {
    outer_prefix: i32,
    inner_prefix: i32,
    outer_suffix: i32,
}

/// `org.apache.lucene.analysis.hunspell.Stemmer`.
#[derive(Debug, Clone, Copy)]
pub struct Stemmer<'d> {
    pub(crate) dictionary: &'d Dictionary,
    form_step: usize,
}

impl<'d> Stemmer<'d> {
    /// `new Stemmer(dictionary)`.
    pub fn new(dictionary: &'d Dictionary) -> Self {
        Stemmer {
            dictionary,
            form_step: dictionary.form_step(),
        }
    }

    /// `stem(String)`: every root of `word`, in Lucene's order.
    pub fn stem(&self, word: &str) -> Vec<String> {
        let units: Vec<u16> = word.encode_utf16().collect();
        self.stem_units(&units)
            .iter()
            .map(|s| String::from_utf16_lossy(s))
            .collect()
    }

    /// `uniqueStems(char[], int)`: [`Self::stem`] without repeats (compared
    /// ignoring case when the dictionary ignores case).
    pub fn unique_stems(&self, word: &str) -> Vec<String> {
        let units: Vec<u16> = word.encode_utf16().collect();
        self.unique_stems_units(&units)
            .iter()
            .map(|s| String::from_utf16_lossy(s))
            .collect()
    }

    /// `stem(char[], int)` over UTF-16 units.
    pub(crate) fn stem_units(&self, word: &[u16]) -> Vec<Vec<u16>> {
        let mut list = Vec::new();
        self.analyze(word, &mut |m| {
            list.push(self.new_stem(m.stem, m.morph_data_id));
            true
        });
        list
    }

    /// `uniqueStems` over UTF-16 units: `CharArraySet`'s case-insensitive
    /// comparison is `Character.toLowerCase` per code point.
    pub(crate) fn unique_stems_units(&self, word: &[u16]) -> Vec<Vec<u16>> {
        let stems = self.stem_units(word);
        if stems.len() < 2 {
            return stems;
        }
        let key = |s: &[u16]| -> Vec<u32> {
            char::decode_utf16(s.iter().copied())
                .map(|c| c.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from))
                .map(|cp| {
                    if self.dictionary.ignore_case {
                        crate::java_character::to_lower_case(cp)
                    } else {
                        cp
                    }
                })
                .collect()
        };
        // A word has a handful of stems: a linear scan beats hashing.
        let mut seen: Vec<Vec<u32>> = Vec::with_capacity(stems.len());
        stems
            .into_iter()
            .filter(|s| {
                let k = key(s);
                if seen.contains(&k) {
                    false
                } else {
                    seen.push(k);
                    true
                }
            })
            .collect()
    }

    /// `analyze`: input cleaning, the word itself, then its case variants.
    pub(crate) fn analyze(&self, word: &[u16], processor: &mut dyn FnMut(&RootMatch<'_>) -> bool) {
        let d = self.dictionary;
        let cleaned;
        let mut word = word;
        if d.may_need_input_cleaning() && d.needs_input_cleaning(word) {
            cleaned = d.clean_input(word);
            word = &cleaned;
        }
        if word.is_empty() {
            return;
        }
        if !self.do_stem(word, WordContext::SimpleWord, processor) {
            return;
        }
        let case = self.case_of(word);
        if case == WordCase::Upper || case == WordCase::Title {
            self.vary_case(word, case, &mut |variant, _| {
                self.do_stem(variant, WordContext::SimpleWord, processor)
            });
        }
    }

    /// `caseOf`: `MIXED` when case is ignored or the word starts lower-case.
    pub(crate) fn case_of(&self, word: &[u16]) -> WordCase {
        if self.dictionary.ignore_case || word.is_empty() || super::word_case::is_lower(word[0]) {
            return WordCase::Mixed;
        }
        WordCase::case_of(word)
    }

    /// `varyCase`: the title-case, apostrophe-capitalized, sharp-s and
    /// lower-case variants of an upper- or title-case word.
    pub(crate) fn vary_case(
        &self,
        word: &[u16],
        case: WordCase,
        processor: &mut dyn FnMut(&[u16], Option<WordCase>) -> bool,
    ) -> bool {
        let d = self.dictionary;
        let title = if case == WordCase::Upper {
            let mut t = word.to_vec();
            for c in t.iter_mut().skip(1) {
                *c = d.case_fold(*c);
            }
            Some(t)
        } else {
            None
        };
        if let Some(title) = &title {
            if let Some(apos) = capitalize_after_apostrophe(title) {
                if !processor(&apos, Some(case)) {
                    return false;
                }
            }
            if !processor(title, Some(case)) {
                return false;
            }
            if d.check_sharp_s && !vary_sharp_s(title, processor) {
                return false;
            }
        }
        if d.is_dot_i_case_change_disallowed(word) {
            return true;
        }
        let mut lower = title.clone().unwrap_or_else(|| word.to_vec());
        lower[0] = d.case_fold(lower[0]);
        if !processor(&lower, Some(case)) {
            return false;
        }
        if case == WordCase::Upper && d.check_sharp_s && !vary_sharp_s(&lower, processor) {
            return false;
        }
        true
    }

    /// `doStem`: `word` itself if it is a root, then its affix removals.
    pub(crate) fn do_stem(
        &self,
        word: &[u16],
        context: WordContext,
        processor: &mut dyn FnMut(&RootMatch<'_>) -> bool,
    ) -> bool {
        let d = self.dictionary;
        if let Some(forms) = d.lookup_word(word) {
            for i in (0..forms.len()).step_by(self.form_step) {
                let entry_id = forms[i];
                if d.has_flag(entry_id, d.needaffix) {
                    continue;
                }
                if (context == WordContext::CompoundBegin || context == WordContext::CompoundMiddle)
                    && d.has_flag(entry_id, d.compound_forbid)
                {
                    return false;
                }
                if !self.is_root_compatible_with_context(context, -1, entry_id) {
                    continue;
                }
                let m = RootMatch {
                    stem: word,
                    form_id: entry_id,
                    morph_data_id: self.morph_data_id(forms, i),
                    outer_prefix: -1,
                    inner_prefix: -1,
                    outer_suffix: -1,
                    inner_suffix: -1,
                };
                if !processor(&m) {
                    return false;
                }
            }
        }
        let applied = Applied {
            outer_prefix: -1,
            inner_prefix: -1,
            outer_suffix: -1,
        };
        self.remove_affixes(word, true, applied, context, &mut |c| {
            self.process_stem_candidate(
                c.word,
                c.last_affix,
                c.outer_prefix,
                c.inner_prefix,
                c.outer_suffix,
                c.inner_suffix,
                context,
                processor,
            )
        })
    }

    /// `removeAffixes(word, 0, length, true, -1, -1, -1, processor)` with a
    /// processor seeing each stripped candidate (`WordFormGenerator`).
    pub(crate) fn remove_affixes_for_candidates(
        &self,
        word: &[u16],
        f: &mut dyn FnMut(&[u16]) -> bool,
    ) -> bool {
        self.remove_affixes_with_candidates(word, &mut |c| f(c.word))
    }

    /// [`Self::remove_affixes_for_candidates`] with the removed affixes
    /// (`WordFormGenerator.compress`).
    pub(crate) fn remove_affixes_with_candidates(
        &self,
        word: &[u16],
        f: &mut dyn FnMut(&Candidate<'_>) -> bool,
    ) -> bool {
        let applied = Applied {
            outer_prefix: -1,
            inner_prefix: -1,
            outer_suffix: -1,
        };
        self.remove_affixes(word, true, applied, WordContext::SimpleWord, f)
    }

    /// The `StemCandidateProcessor` of `doStem`: a stripped word whose root
    /// carries the removed affix's flag (and the outer prefix's).
    #[allow(clippy::too_many_arguments)]
    fn process_stem_candidate(
        &self,
        word: &[u16],
        last_affix: i32,
        outer_prefix: i32,
        inner_prefix: i32,
        outer_suffix: i32,
        inner_suffix: i32,
        context: WordContext,
        processor: &mut dyn FnMut(&RootMatch<'_>) -> bool,
    ) -> bool {
        let d = self.dictionary;
        let Some(forms) = d.lookup_word(word) else {
            return true;
        };
        let flag = d.affix_data(last_affix, AFFIX_FLAG);
        let prefix_id = if inner_prefix >= 0 {
            inner_prefix
        } else {
            outer_prefix
        };
        for i in (0..forms.len()).step_by(self.form_step) {
            let entry_id = forms[i];
            if d.has_flag(entry_id, flag) || d.is_flag_appended_by_affix(prefix_id, flag) {
                if inner_prefix < 0 && outer_prefix >= 0 {
                    let prefix_flag = d.affix_data(outer_prefix, AFFIX_FLAG);
                    if !d.has_flag(entry_id, prefix_flag)
                        && !d.is_flag_appended_by_affix(last_affix, prefix_flag)
                    {
                        continue;
                    }
                }
                if !self.is_root_compatible_with_context(context, last_affix, entry_id) {
                    continue;
                }
                let m = RootMatch {
                    stem: word,
                    form_id: entry_id,
                    morph_data_id: self.morph_data_id(forms, i),
                    outer_prefix,
                    inner_prefix,
                    outer_suffix,
                    inner_suffix,
                };
                if !processor(&m) {
                    return false;
                }
            }
        }
        true
    }

    /// `stemException`: the `st:` field of the entry's morphological data.
    fn stem_exception(&self, morph_data_id: i32) -> Option<Vec<u16>> {
        if morph_data_id <= 0 {
            return None;
        }
        const ST: [u16; 3] = [b's' as u16, b't' as u16, b':' as u16];
        const SPACE_ST: [u16; 4] = [b' ' as u16, b's' as u16, b't' as u16, b':' as u16];
        let data = &self.dictionary.morph_data[morph_data_id as usize];
        let start = if data.starts_with(&ST) {
            0
        } else {
            super::dictionary::index_of_str(data, &SPACE_ST, 0)?
        };
        let next_space = super::dictionary::index_of(data, u16::from(b' '), start + 3);
        Some(data[start + 3..next_space.unwrap_or(data.len())].to_vec())
    }

    /// `newStem`: the stem exception if any, through `OCONV`.
    pub(crate) fn new_stem(&self, stem: &[u16], morph_data_id: i32) -> Vec<u16> {
        let mut out = self
            .stem_exception(morph_data_id)
            .unwrap_or_else(|| stem.to_vec());
        if let Some(oconv) = &self.dictionary.oconv {
            oconv.apply_mappings(&mut out);
        }
        out
    }

    /// `removeAffixes`.
    fn remove_affixes(
        &self,
        word: &[u16],
        do_prefix: bool,
        applied: Applied,
        context: WordContext,
        processor: &mut dyn FnMut(&Candidate<'_>) -> bool,
    ) -> bool {
        let d = self.dictionary;
        let length = word.len();
        if do_prefix {
            let fst = &d.prefixes;
            let mut node = Some(fst.root());
            let limit = if d.full_strip { length + 1 } else { length };
            for i in 0..limit {
                if i > 0 {
                    node = node.and_then(|n| fst.step(n, word[i - 1]));
                    if node.is_none() {
                        break;
                    }
                }
                let Some(prefixes) = node.and_then(|n| fst.finals(n)) else {
                    continue;
                };
                for &prefix in prefixes {
                    if prefix == applied.outer_prefix {
                        continue;
                    }
                    if self.is_affix_compatible(
                        prefix,
                        true,
                        applied.outer_prefix,
                        applied.outer_suffix,
                        context,
                    ) {
                        let applied_ok = self.strip_affix(word, i, prefix, true, &mut |stripped| {
                            self.apply_affix(stripped, prefix, true, applied, context, processor)
                        });
                        if applied_ok == Some(false) {
                            return false;
                        }
                    }
                }
            }
        }
        let fst = &d.suffixes;
        let mut node = Some(fst.root());
        let limit = if d.full_strip { 0 } else { 1 };
        let mut i = length;
        loop {
            if i < limit {
                break;
            }
            if i < length {
                node = node.and_then(|n| fst.step(n, word[i]));
                if node.is_none() {
                    break;
                }
            }
            if let Some(suffixes) = node.and_then(|n| fst.finals(n)) {
                for &suffix in suffixes {
                    if suffix == applied.outer_suffix {
                        continue;
                    }
                    if self.is_affix_compatible(
                        suffix,
                        false,
                        applied.outer_prefix,
                        applied.outer_suffix,
                        context,
                    ) {
                        let applied_ok =
                            self.strip_affix(word, length - i, suffix, false, &mut |stripped| {
                                self.apply_affix(
                                    stripped, suffix, false, applied, context, processor,
                                )
                            });
                        if applied_ok == Some(false) {
                            return false;
                        }
                    }
                }
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
        true
    }

    /// `stripAffix`: hands `then` the word with the affix removed and the
    /// strip restored and returns its answer, or `None` when the condition
    /// fails or nothing would remain.
    fn strip_affix(
        &self,
        word: &[u16],
        affix_len: usize,
        affix: i32,
        is_prefix: bool,
        then: &mut dyn FnMut(&[u16]) -> bool,
    ) -> Option<bool> {
        let d = self.dictionary;
        let de_affixed_len = word.len() - affix_len;
        let strip_ord = usize::from(d.affix_data(affix, AFFIX_STRIP_ORD));
        let strip = &d.strip_data[d.strip_offsets[strip_ord]..d.strip_offsets[strip_ord + 1]];
        if strip.len() + de_affixed_len == 0 {
            return None;
        }
        let de_affixed = if is_prefix {
            &word[affix_len..]
        } else {
            &word[..de_affixed_len]
        };
        let condition = d.get_affix_condition(affix);
        if condition != 0 {
            if let Some(Some(c)) = d.patterns.get(condition) {
                if !c.accepts_stem(de_affixed) {
                    return None;
                }
            }
        }
        // The stripped word, on the stack when it fits (Java allocates a
        // `char[]`; here that was a malloc per candidate).
        let (first, second) = if is_prefix {
            (strip, de_affixed)
        } else {
            (de_affixed, strip)
        };
        let total = first.len() + second.len();
        let mut inline = [0u16; 64];
        if total <= inline.len() {
            inline[..first.len()].copy_from_slice(first);
            inline[first.len()..total].copy_from_slice(second);
            Some(then(&inline[..total]))
        } else {
            Some(then(&[first, second].concat()))
        }
    }

    /// `isAffixCompatible`.
    fn is_affix_compatible(
        &self,
        affix: i32,
        is_prefix: bool,
        outer_prefix: i32,
        outer_suffix: i32,
        context: WordContext,
    ) -> bool {
        let d = self.dictionary;
        let append = i32::from(d.affix_data(affix, AFFIX_APPEND));
        let previous_was_prefix = outer_suffix < 0 && outer_prefix >= 0;
        if context.is_compound() {
            if !is_prefix && d.has_flag(append, d.compound_forbid) {
                return false;
            }
            if !context.is_affix_allowed_without_special_permit(is_prefix)
                && !d.has_flag(append, d.compound_permit)
            {
                return false;
            }
            if context == WordContext::CompoundEnd
                && !is_prefix
                && !previous_was_prefix
                && d.has_flag(append, d.onlyincompound)
            {
                return false;
            }
        } else if d.has_flag(append, d.onlyincompound) {
            return false;
        }
        if outer_prefix == -1 && outer_suffix == -1 {
            return true;
        }
        if d.is_cross_product(affix) {
            if previous_was_prefix {
                return true;
            }
            if outer_suffix >= 0 {
                let prev_flag = d.affix_data(outer_suffix, AFFIX_FLAG);
                return d.has_flag(append, prev_flag);
            }
        }
        false
    }

    /// `applyAffix`.
    fn apply_affix(
        &self,
        word: &[u16],
        affix: i32,
        prefix: bool,
        applied: Applied,
        context: WordContext,
        processor: &mut dyn FnMut(&Candidate<'_>) -> bool,
    ) -> bool {
        let d = self.dictionary;
        let Applied {
            mut outer_prefix,
            mut inner_prefix,
            mut outer_suffix,
        } = applied;
        let prefix_id = if inner_prefix >= 0 {
            inner_prefix
        } else {
            outer_prefix
        };
        let previous_affix = if outer_suffix >= 0 {
            outer_suffix
        } else {
            prefix_id
        };
        let mut inner_suffix = -1;
        if prefix {
            if outer_prefix < 0 {
                outer_prefix = affix;
            } else {
                inner_prefix = affix;
            }
        } else if outer_suffix < 0 {
            outer_suffix = affix;
        } else {
            inner_suffix = affix;
        }
        let skip_lookup = self.needs_another_affix(affix, previous_affix, !prefix, prefix_id);
        if !skip_lookup
            && !processor(&Candidate {
                word,
                last_affix: affix,
                outer_prefix,
                inner_prefix,
                outer_suffix,
                inner_suffix,
            })
        {
            return false;
        }
        if inner_suffix >= 0 {
            return true;
        }
        let recursion_depth = i32::from(outer_suffix >= 0)
            + if inner_prefix >= 0 {
                2
            } else {
                i32::from(outer_prefix >= 0)
            }
            - 1;
        if d.is_cross_product(affix) && recursion_depth <= 1 {
            let flag = d.affix_data(affix, AFFIX_FLAG);
            let do_prefix;
            if recursion_depth == 0 {
                if prefix {
                    do_prefix = d.complex_prefixes && d.is_second_stage_prefix(flag);
                } else if !d.complex_prefixes && d.is_second_stage_suffix(flag) {
                    do_prefix = false;
                } else {
                    return true;
                }
            } else if prefix && d.complex_prefixes {
                do_prefix = true;
            } else if prefix || d.complex_prefixes || !d.is_second_stage_suffix(flag) {
                return true;
            } else {
                do_prefix = false;
            }
            let next = Applied {
                outer_prefix,
                inner_prefix,
                outer_suffix,
            };
            return self.remove_affixes(word, do_prefix, next, context, processor);
        }
        true
    }

    /// `isRootCompatibleWithContext`.
    fn is_root_compatible_with_context(
        &self,
        context: WordContext,
        last_affix: i32,
        entry_id: i32,
    ) -> bool {
        let d = self.dictionary;
        if !context.is_compound() && d.has_flag(entry_id, d.onlyincompound) {
            return false;
        }
        if context.is_compound() && context != WordContext::CompoundRuleEnd {
            let c_flag = context.required_flag(d);
            return d.has_flag(entry_id, c_flag)
                || d.is_flag_appended_by_affix(last_affix, c_flag)
                || d.has_flag(entry_id, d.compound_flag)
                || d.is_flag_appended_by_affix(last_affix, d.compound_flag);
        }
        true
    }

    fn morph_data_id(&self, forms: &[i32], i: usize) -> i32 {
        if self.dictionary.has_custom_morph_data {
            forms[i + 1]
        } else {
            0
        }
    }

    /// `needsAnotherAffix`: circumfix and needaffix pairing.
    fn needs_another_affix(
        &self,
        affix: i32,
        previous_affix: i32,
        is_suffix: bool,
        prefix_id: i32,
    ) -> bool {
        let d = self.dictionary;
        if is_suffix
            && d.is_flag_appended_by_affix(prefix_id, d.circumfix)
                != d.is_flag_appended_by_affix(affix, d.circumfix)
        {
            return true;
        }
        if d.is_flag_appended_by_affix(affix, d.needaffix) {
            return !is_suffix
                || previous_affix < 0
                || d.is_flag_appended_by_affix(previous_affix, d.needaffix);
        }
        false
    }
}

/// `Stemmer.capitalizeAfterApostrophe` (Catalan, French, Italian:
/// `SANT'ELIA` -> `Sant'Elia`).
fn capitalize_after_apostrophe(word: &[u16]) -> Option<Vec<u16>> {
    for i in 1..word.len().saturating_sub(1) {
        if word[i] == u16::from(b'\'') {
            let next = word[i + 1];
            let upper = to_upper(next);
            if upper != next {
                let mut copy = word.to_vec();
                copy[i + 1] = to_upper(upper);
                return Some(copy);
            }
        }
    }
    None
}

/// `Stemmer.varySharpS`: every way of spelling each `ss` as `ß` (up to
/// five levels deep), each but the word itself passed to `processor`.
fn vary_sharp_s(word: &[u16], processor: &mut dyn FnMut(&[u16], Option<WordCase>) -> bool) -> bool {
    let s = u16::from(b's');
    let find_ss = |start: usize| {
        (start..word.len().saturating_sub(1)).find(|&i| word[i] == s && word[i + 1] == s)
    };
    fn replace(
        word: &[u16],
        start: usize,
        depth: usize,
        find_ss: &dyn Fn(usize) -> Option<usize>,
    ) -> Option<Vec<Vec<u16>>> {
        if depth > 5 {
            return Some(vec![word[start..].to_vec()]);
        }
        let ss = find_ss(start)?;
        let prefix = &word[start..ss];
        let tails = replace(word, ss + 2, depth + 1, find_ss)
            .unwrap_or_else(|| vec![word[ss + 2..].to_vec()]);
        let mut out = Vec::with_capacity(tails.len() * 2);
        for t in tails {
            out.push([prefix, &[0x73, 0x73], &t].concat());
            out.push([prefix, &[0x00DF], &t].concat());
        }
        Some(out)
    }
    let Some(variants) = replace(word, 0, 0, &find_ss) else {
        return true;
    };
    for v in variants {
        if v != word && !processor(&v, None) {
            return false;
        }
    }
    true
}
