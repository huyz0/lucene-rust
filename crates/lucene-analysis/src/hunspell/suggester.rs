//! Suggestions: `Suggester`, `ModifyingSuggester` (edits of the
//! misspelling), `GeneratingSuggester` (dictionary roots ranked by n-gram
//! similarity, expanded with affixes), `TrigramAutomaton` and `Suggestion`.
//!
//! Differs:
//! - `TrigramAutomaton.ngramScore` (the sum, over the distinct substrings of
//!   up to three units of a root, of their occurrences in the misspelling)
//!   is computed from a map of the misspelling's substrings instead of a
//!   determinized automaton; the score is the same.
//! - The suggestion speller's `findStem` cache is not kept (it only saves
//!   repeated lookups).
//! - `String.toUpperCase(Locale.ROOT)` maps per code point plus `ß` -> `SS`;
//!   the JDK's other unconditional special-casing expansions (ligatures,
//!   `ŉ`, Greek with ypogegrammeni) map by their simple mapping.
//! - `suggest("")` answers nothing where Java throws
//!   `StringIndexOutOfBoundsException`.
//! - Only `TimeoutPolicy.NO_TIMEOUT`; no `SuggestibleEntryCache` (a
//!   speed-for-memory option that changes no suggestion).

use std::collections::{BTreeSet, HashMap, HashSet};

use super::dictionary::{index_of_str, AFFIX_APPEND, AFFIX_FLAG, AFFIX_STRIP_ORD};
use super::speller::Hunspell;
use super::stemmer::Stemmer;
use super::word_case::{is_upper, to_upper, WordCase};
use super::word_form_generator::{EverythingPossible, FragmentChecker};
use super::Dictionary;
use super::FLAG_UNSET;
use crate::java_character as jc;

/// `String.toUpperCase(Locale.ROOT)` (see the module docs).
pub(crate) fn java_upper(s: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(s.len());
    for c in char::decode_utf16(s.iter().copied()) {
        match c {
            Ok('ß') => out.extend_from_slice(&[u16::from(b'S'), u16::from(b'S')]),
            Ok(c) => {
                jc::push_utf16(&mut out, jc::to_upper_case(c as u32));
            }
            Err(e) => out.push(e.unpaired_surrogate()),
        }
    }
    out
}

fn contains_sub(hay: &[u16], needle: &[u16]) -> bool {
    index_of_str(hay, needle, 0).is_some()
}

/// `Suggestion`: a raw candidate and what it is reported as.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Suggestion {
    pub(crate) raw: Vec<u16>,
    result: Vec<Vec<u16>>,
}

impl Suggestion {
    fn new(
        raw: Vec<u16>,
        misspelled: &[u16],
        case: WordCase,
        speller: &Hunspell<'_>,
    ) -> Suggestion {
        let adjusted = adjust_suggestion_case(&raw, misspelled, case);
        let chosen = if adjusted.contains(&u16::from(b' ')) || speller.spell_units(&adjusted) {
            adjusted
        } else {
            raw.clone()
        };
        let mut result = vec![clean_output(speller, chosen)];
        if case == WordCase::Upper && speller.dictionary.check_sharp_s && raw.contains(&0x00DF) {
            result.push(clean_output(speller, raw.clone()));
        }
        Suggestion { raw, result }
    }
}

fn adjust_suggestion_case(candidate: &[u16], misspelled: &[u16], case: WordCase) -> Vec<u16> {
    if case == WordCase::Upper {
        return java_upper(candidate);
    }
    if is_upper(misspelled[0]) && !candidate.is_empty() {
        let mut c = candidate.to_vec();
        c[0] = to_upper(c[0]);
        return c;
    }
    candidate.to_vec()
}

fn clean_output(speller: &Hunspell<'_>, mut s: Vec<u16>) -> Vec<u16> {
    if let Some(oconv) = &speller.dictionary.oconv {
        oconv.apply_mappings(&mut s);
    }
    s
}

/// An insertion-ordered set of suggestions (`LinkedHashSet<Suggestion>`).
#[derive(Default)]
pub(crate) struct SuggestionSet {
    items: Vec<Suggestion>,
    seen: HashSet<Suggestion>,
}

impl SuggestionSet {
    fn add(&mut self, s: Suggestion) -> bool {
        if self.seen.insert(s.clone()) {
            self.items.push(s);
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
        self.seen.clear();
    }

    fn take(&mut self) -> Vec<Suggestion> {
        self.seen.clear();
        std::mem::take(&mut self.items)
    }
}

impl Hunspell<'_> {
    /// `suggest(word)` under `TimeoutPolicy.NO_TIMEOUT`: `new
    /// Suggester(dictionary)`'s suggestions.
    pub fn suggest(&self, word: &str) -> Vec<String> {
        Suggester::new(self.dictionary).suggest_no_timeout(word)
    }
}

/// `org.apache.lucene.analysis.hunspell.Suggester`: suggestions for a
/// misspelled word, with Lucene's tuning options.
#[derive(Clone, Copy)]
pub struct Suggester<'d> {
    dictionary: &'d Dictionary,
    fragment_checker: &'d dyn FragmentChecker,
    proceed_past_rep: bool,
}

impl<'d> Suggester<'d> {
    /// `new Suggester(dictionary)`: `FragmentChecker.EVERYTHING_POSSIBLE`,
    /// stopping after a `REP` suggestion.
    pub fn new(dictionary: &'d Dictionary) -> Self {
        Suggester {
            dictionary,
            fragment_checker: &EverythingPossible,
            proceed_past_rep: false,
        }
    }

    /// `withFragmentChecker(checker)`: skip modifications that introduce an
    /// impossible fragment.
    pub fn with_fragment_checker(self, checker: &'d dyn FragmentChecker) -> Self {
        Suggester {
            fragment_checker: checker,
            ..self
        }
    }

    /// `proceedPastRep()`: keep trying modifications after a `REP` rule gave
    /// an acceptable word.
    pub fn proceed_past_rep(self) -> Self {
        Suggester {
            proceed_past_rep: true,
            ..self
        }
    }

    /// `suggestNoTimeout(word, () -> {})`.
    pub fn suggest_no_timeout(&self, word: &str) -> Vec<String> {
        let units: Vec<u16> = word.encode_utf16().collect();
        self.suggest_units(&units)
            .iter()
            .map(|s| String::from_utf16_lossy(s))
            .collect()
    }

    /// `Suggester.suggest` over UTF-16 units.
    pub(crate) fn suggest_units(&self, word: &[u16]) -> Vec<Vec<u16>> {
        let d = self.dictionary;
        if word.len() >= 100 || word.is_empty() {
            return vec![];
        }
        let cleaned;
        let mut word = word;
        if d.needs_input_cleaning(word) {
            cleaned = d.clean_input(word);
            word = &cleaned;
        }
        if word.is_empty() {
            return vec![];
        }
        let speller = Hunspell {
            dictionary: d,
            stemmer: Stemmer::new(d),
            suggestion_mode: true,
        };
        let case = WordCase::case_of(word);
        if d.force_u_case != FLAG_UNSET && case == WordCase::Lower {
            let title = d.to_title_case(word);
            if speller.spell_units(&title) {
                return vec![title];
            }
        }
        let mut suggestions = SuggestionSet::default();
        let has_good = ModifyingSuggester {
            speller: &speller,
            result: &mut suggestions,
            misspelled: word,
            word_case: case,
            tried: HashSet::new(),
            fragment_checker: self.fragment_checker,
            proceed_past_rep: self.proceed_past_rep,
        }
        .suggest();
        if !has_good && d.max_ngram_suggestions > 0 {
            let lower = d.to_lower_case(word);
            let generated =
                GeneratingSuggester { speller: &speller }.suggest(&lower, case, &suggestions);
            for raw in generated {
                suggestions.add(Suggestion::new(raw, word, case, &speller));
            }
        }
        let dash = u16::from(b'-');
        if word.contains(&dash) && !suggestions.items.iter().any(|s| s.raw.contains(&dash)) {
            for raw in self.modify_chunks_between_dashes(word, &speller) {
                suggestions.add(Suggestion::new(raw, word, case, &speller));
            }
        }
        postprocess(&suggestions.items)
    }

    /// `Suggester.modifyChunksBetweenDashes`.
    fn modify_chunks_between_dashes(&self, word: &[u16], speller: &Hunspell<'_>) -> Vec<Vec<u16>> {
        let mut result = Vec::new();
        let mut chunk_start = 0;
        while chunk_start < word.len() {
            let chunk_end = super::dictionary::index_of(word, u16::from(b'-'), chunk_start)
                .unwrap_or(word.len());
            if chunk_end > chunk_start {
                let chunk = &word[chunk_start..chunk_end];
                if !speller.spell_units(chunk) {
                    for sug in self.suggest_units(chunk) {
                        let replaced =
                            [&word[..chunk_start], sug.as_slice(), &word[chunk_end..]].concat();
                        if speller.spell_units(&replaced) {
                            result.push(replaced);
                        }
                    }
                }
            }
            chunk_start = chunk_end + 1;
        }
        result
    }
}

/// `Suggester.postprocess`: every suggestion's results, deduplicated.
fn postprocess(suggestions: &[Suggestion]) -> Vec<Vec<u16>> {
    let mut out: Vec<Vec<u16>> = Vec::new();
    for s in suggestions {
        for r in &s.result {
            if !out.contains(r) {
                out.push(r.clone());
            }
        }
    }
    out
}

/// `ModifyingSuggester.MAX_CHAR_DISTANCE`.
const MAX_CHAR_DISTANCE: usize = 4;

/// `ModifyingSuggester`.
struct ModifyingSuggester<'a, 'd> {
    speller: &'a Hunspell<'d>,
    result: &'a mut SuggestionSet,
    misspelled: &'a [u16],
    word_case: WordCase,
    tried: HashSet<Vec<u16>>,
    fragment_checker: &'a dyn FragmentChecker,
    proceed_past_rep: bool,
}

#[derive(PartialEq, Eq)]
enum Graded {
    None,
    Normal,
    Best,
}

impl ModifyingSuggester<'_, '_> {
    fn create(&self, candidate: Vec<u16>) -> Suggestion {
        Suggestion::new(candidate, self.misspelled, self.word_case, self.speller)
    }

    fn suggest(&mut self) -> bool {
        let d = self.speller.dictionary;
        let misspelled = self.misspelled.to_vec();
        let low = if self.word_case != WordCase::Lower {
            d.to_lower_case(&misspelled)
        } else {
            misspelled.clone()
        };
        if self.word_case == WordCase::Upper || self.word_case == WordCase::Mixed {
            self.try_suggestion(low.clone());
        }
        let mut has_good = self.try_variations_of(&misspelled);
        match self.word_case {
            WordCase::Title => has_good |= self.try_variations_of(&low),
            WordCase::Upper => {
                has_good |= self.try_variations_of(&low);
                has_good |= self.try_variations_of(&d.to_title_case(&misspelled));
            }
            WordCase::Mixed => {
                let dot = super::dictionary::index_of(&misspelled, u16::from(b'.'), 0);
                if let Some(dot) = dot {
                    if dot > 0 && dot < misspelled.len() - 1 {
                        let after = &misspelled[dot + 1..];
                        if WordCase::case_of(after) == WordCase::Title {
                            let s = [&misspelled[..=dot], &[u16::from(b' ')][..], after].concat();
                            let sug = self.create(s);
                            self.result.add(sug);
                        }
                    }
                }
                let first = misspelled[0];
                let capitalized = is_upper(first);
                if capitalized {
                    let mut v = misspelled.clone();
                    v[0] = d.case_fold(first);
                    has_good |= self.try_variations_of(&v);
                }
                has_good |= self.try_variations_of(&low);
                if capitalized {
                    has_good |= self.try_variations_of(&d.to_title_case(&low));
                }
                let mut reordered: Vec<Suggestion> = Vec::new();
                for candidate in self.result.take() {
                    match self.capitalize_after_space(&candidate.raw) {
                        None => reordered.push(candidate),
                        Some(changed) => reordered.insert(0, changed),
                    }
                }
                for s in reordered {
                    self.result.add(s);
                }
            }
            WordCase::Lower | WordCase::Neutral => {}
        }
        has_good
    }

    /// `capitalizeAfterSpace`: `aNew` -> `a New`.
    fn capitalize_after_space(&self, candidate: &[u16]) -> Option<Suggestion> {
        let space = super::dictionary::index_of(candidate, u16::from(b' '), 0)?;
        if space == 0 || space + 1 >= candidate.len() {
            return None;
        }
        let tail = candidate.len() - space - 1;
        let m = self.misspelled;
        let region_matches = m.len() >= tail && m[m.len() - tail..] == candidate[space + 1..];
        if region_matches {
            return None;
        }
        let mut s = candidate[..=space].to_vec();
        s.push(to_upper(candidate[space + 1]));
        s.extend_from_slice(&candidate[space + 2..]);
        Some(self.create(s))
    }

    fn try_variations_of(&mut self, word: &[u16]) -> bool {
        let d = self.speller.dictionary;
        let mut has_good = self.try_suggestion(java_upper(word));
        let rep = self.try_rep(word);
        if rep == Graded::Best && !self.proceed_past_rep {
            return true;
        }
        has_good |= rep != Graded::None;
        if !d.map_table.is_empty() {
            self.enumerate_map_replacements(word, Vec::new(), 0);
        }
        self.try_swapping_chars(word);
        self.try_long_swap(word);
        self.try_neighbor_keys(word);
        self.try_removing_char(word);
        self.try_adding_char(word);
        self.try_moving_char(word);
        self.try_replacing_char(word);
        self.try_two_duplicate_chars(word);
        let good_split = self.check_dictionary_for_split_suggestions(word);
        if !good_split.is_empty() {
            let copy = self.result.take();
            self.result.clear();
            for s in good_split {
                self.result.add(s);
            }
            if has_good {
                for s in copy {
                    self.result.add(s);
                }
            }
            has_good = true;
        }
        if !has_good && d.enable_split_suggestions {
            self.try_splitting(word);
        }
        has_good
    }

    fn try_rep(&mut self, word: &[u16]) -> Graded {
        let d = self.speller.dictionary;
        let mut has_best = false;
        let before = self.result.len();
        for entry in &d.rep_table {
            for candidate in entry.substitute(word) {
                let candidate = super::dictionary::java_trim(&candidate).to_vec();
                if self.try_suggestion(candidate.clone()) {
                    has_best = true;
                    continue;
                }
                let space = u16::from(b' ');
                if candidate.contains(&space)
                    && split_space(&candidate)
                        .iter()
                        .all(|p| self.check_simple_word(p))
                {
                    let s = self.create(candidate);
                    self.result.add(s);
                }
            }
        }
        if has_best {
            Graded::Best
        } else if self.result.len() > before {
            Graded::Normal
        } else {
            Graded::None
        }
    }

    fn enumerate_map_replacements(&mut self, word: &[u16], accumulated: Vec<u16>, offset: usize) {
        if offset == word.len() {
            self.try_suggestion(accumulated);
            return;
        }
        let d = self.speller.dictionary;
        let length = accumulated.len();
        for entries in &d.map_table {
            for entry in entries {
                if word.len() >= offset + entry.len()
                    && word[offset..offset + entry.len()] == entry[..]
                {
                    for replacement in entries {
                        if entry != replacement {
                            let next = [accumulated.as_slice(), replacement.as_slice()].concat();
                            let end = length + replacement.len();
                            if !self
                                .fragment_checker
                                .has_impossible_fragment_around(&next, length, end)
                            {
                                self.enumerate_map_replacements(word, next, offset + entry.len());
                            }
                        }
                    }
                }
            }
        }
        let mut next = accumulated;
        next.push(word[offset]);
        if !self
            .fragment_checker
            .has_impossible_fragment_around(&next, length, length + 1)
        {
            self.enumerate_map_replacements(word, next, offset + 1);
        }
    }

    fn check_simple_word(&self, part: &[u16]) -> bool {
        self.speller.check_simple_word(part, None) == Some(true)
    }

    fn try_swapping_chars(&mut self, word: &[u16]) {
        let length = word.len();
        for i in 0..length.saturating_sub(1) {
            let mut c = word.to_vec();
            c.swap(i, i + 1);
            self.try_suggestion(c);
        }
        if length == 4 || length == 5 {
            let mut candidate = word.to_vec();
            candidate[0] = word[1];
            candidate[1] = word[0];
            candidate[length - 1] = word[length - 2];
            candidate[length - 2] = word[length - 1];
            self.try_suggestion(candidate.clone());
            if length == 5 {
                candidate[0] = word[0];
                candidate[1] = word[2];
                candidate[2] = word[1];
                self.try_suggestion(candidate);
            }
        }
    }

    fn try_neighbor_keys(&mut self, word: &[u16]) {
        let d = self.speller.dictionary;
        for i in 0..word.len() {
            let c = word[i];
            let up = to_upper(c);
            if up != c {
                let mut s = word.to_vec();
                s[i] = up;
                self.try_suggestion(s);
            }
            for group in &d.neighbor_key_groups {
                if group.contains(&c) {
                    for &g in group {
                        if g != c {
                            let mut s = word.to_vec();
                            s[i] = g;
                            self.try_modified_suggestions(i, s);
                        }
                    }
                }
            }
        }
    }

    fn try_modified_suggestions(&mut self, mod_offset: usize, candidate: Vec<u16>) {
        if !self.fragment_checker.has_impossible_fragment_around(
            &candidate,
            mod_offset,
            mod_offset + 1,
        ) {
            self.try_suggestion(candidate);
        }
    }

    fn try_long_swap(&mut self, word: &[u16]) {
        for i in 0..word.len() {
            let mut j = i + 2;
            while j < word.len() && j <= i + MAX_CHAR_DISTANCE {
                let mut s = word.to_vec();
                s.swap(i, j);
                self.try_suggestion(s);
                j += 1;
            }
        }
    }

    fn try_removing_char(&mut self, word: &[u16]) {
        if word.len() == 1 {
            return;
        }
        for i in 0..word.len() {
            let s = [&word[..i], &word[i + 1..]].concat();
            self.try_suggestion(s);
        }
    }

    fn try_adding_char(&mut self, word: &[u16]) {
        let try_chars = self.speller.dictionary.try_chars.clone();
        for i in 0..=word.len() {
            for &t in &try_chars {
                let s = [&word[..i], &[t][..], &word[i..]].concat();
                self.try_modified_suggestions(i, s);
            }
        }
    }

    fn try_moving_char(&mut self, word: &[u16]) {
        for i in 0..word.len() {
            let prefix = &word[..i];
            let mut j = i + 2;
            while j < word.len() && j <= i + MAX_CHAR_DISTANCE {
                let a = [prefix, &word[i + 1..j], &[word[i]][..], &word[j..]].concat();
                self.try_suggestion(a);
                let b = [prefix, &[word[j]][..], &word[i..j], &word[j + 1..]].concat();
                self.try_suggestion(b);
                j += 1;
            }
            if i + 1 < word.len() {
                let s = [prefix, &word[i + 1..], &[word[i]][..]].concat();
                self.try_suggestion(s);
            }
        }
    }

    fn try_replacing_char(&mut self, word: &[u16]) {
        let try_chars = self.speller.dictionary.try_chars.clone();
        for i in 0..word.len() {
            for &t in &try_chars {
                if t != word[i] {
                    let mut s = word.to_vec();
                    s[i] = t;
                    self.try_modified_suggestions(i, s);
                }
            }
        }
    }

    fn try_two_duplicate_chars(&mut self, word: &[u16]) {
        let mut dup_len = 0;
        for i in 2..word.len() {
            if word[i] == word[i - 2] {
                dup_len += 1;
                if dup_len == 3 || dup_len == 2 && i >= 4 {
                    let s = [&word[..i - 1], &word[i + 1..]].concat();
                    self.try_suggestion(s);
                    dup_len = 0;
                }
            } else {
                dup_len = 0;
            }
        }
    }

    fn should_split_by_dash(&self) -> bool {
        let t = &self.speller.dictionary.try_chars;
        t.contains(&u16::from(b'-')) || t.contains(&u16::from(b'a'))
    }

    fn check_dictionary_for_split_suggestions(&mut self, word: &[u16]) -> Vec<Suggestion> {
        let mut result = Vec::new();
        for i in 1..word.len().saturating_sub(1) {
            let (w1, w2) = word.split_at(i);
            let spaced = [w1, &[u16::from(b' ')][..], w2].concat();
            if self.speller.check_word(&spaced, None) {
                result.push(self.create(spaced));
            }
            if self.should_split_by_dash() {
                let dashed = [w1, &[u16::from(b'-')][..], w2].concat();
                if self.speller.check_word(&dashed, None) {
                    result.push(self.create(dashed));
                }
            }
        }
        result
    }

    fn try_splitting(&mut self, word: &[u16]) {
        for i in 1..word.len() {
            let (w1, w2) = word.split_at(i);
            if self.check_simple_word(w1) && self.check_simple_word(w2) {
                let s = self.create([w1, &[u16::from(b' ')][..], w2].concat());
                self.result.add(s);
                if w1.len() > 1 && w2.len() > 1 && self.should_split_by_dash() {
                    let s = self.create([w1, &[u16::from(b'-')][..], w2].concat());
                    self.result.add(s);
                }
            }
        }
    }

    fn try_suggestion(&mut self, candidate: Vec<u16>) -> bool {
        if !self.tried.insert(candidate.clone()) || !self.speller.check_word(&candidate, None) {
            return false;
        }
        let s = self.create(candidate);
        self.result.add(s)
    }
}

/// `String.split(" ")`.
fn split_space(s: &[u16]) -> Vec<&[u16]> {
    let mut parts: Vec<&[u16]> = s.split(|&c| c == u16::from(b' ')).collect();
    while parts.len() > 1 && parts.last().is_some_and(|p| p.is_empty()) {
        parts.pop();
    }
    parts
}

const MAX_ROOTS: usize = 100;
const MAX_WORDS: usize = 100;
const MAX_GUESSES: usize = 200;
const MAX_ROOT_LENGTH_DIFF: usize = 4;

/// `GeneratingSuggester.Weighted`, ordered as its `compareTo`: higher scores
/// first, then by the text.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Weighted<T> {
    word: T,
    score: i32,
}

impl<T: Ord> PartialOrd for Weighted<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for Weighted<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .score
            .cmp(&self.score)
            .then_with(|| self.word.cmp(&other.word))
    }
}

/// A root for generation: compared by its text only (`Root.compareTo`).
#[derive(Debug, Clone)]
struct GenRoot {
    word: Vec<u16>,
    entry_id: i32,
}

impl PartialEq for GenRoot {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word
    }
}
impl Eq for GenRoot {}
impl PartialOrd for GenRoot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for GenRoot {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.word.cmp(&other.word)
    }
}

/// `java.util.PriorityQueue` with `Comparator.reverseOrder()` (the head is
/// the greatest), sifting exactly as the JDK does so that ties leave the
/// same element at the head.
struct JavaMaxHeap<T: Ord> {
    queue: Vec<T>,
}

impl<T: Ord> JavaMaxHeap<T> {
    fn new() -> Self {
        JavaMaxHeap { queue: Vec::new() }
    }

    /// `comparator.compare(a, b) < 0` for the reversed comparator.
    fn less(a: &T, b: &T) -> bool {
        b.cmp(a) == std::cmp::Ordering::Less
    }

    fn add(&mut self, x: T) {
        self.queue.push(x);
        let mut k = self.queue.len() - 1;
        while k > 0 {
            let parent = (k - 1) >> 1;
            if !Self::less(&self.queue[k], &self.queue[parent]) {
                break;
            }
            self.queue.swap(k, parent);
            k = parent;
        }
    }

    fn peek(&self) -> Option<&T> {
        self.queue.first()
    }

    fn poll(&mut self) -> Option<T> {
        if self.queue.is_empty() {
            return None;
        }
        let last = self.queue.pop()?;
        if self.queue.is_empty() {
            return Some(last);
        }
        let result = std::mem::replace(&mut self.queue[0], last);
        let n = self.queue.len();
        let mut k = 0;
        let half = n >> 1;
        while k < half {
            let mut child = 2 * k + 1;
            let right = child + 1;
            if right < n && Self::less(&self.queue[right], &self.queue[child]) {
                child = right;
            }
            if !Self::less(&self.queue[child], &self.queue[k]) {
                break;
            }
            self.queue.swap(k, child);
            k = child;
        }
        Some(result)
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

/// `GeneratingSuggester`.
struct GeneratingSuggester<'a, 'd> {
    speller: &'a Hunspell<'d>,
}

impl GeneratingSuggester<'_, '_> {
    fn suggest(&self, word: &[u16], original: WordCase, prev: &SuggestionSet) -> Vec<Vec<u16>> {
        let roots = self.find_similar_dictionary_entries(word, original);
        let expanded = self.expand_roots(word, &roots);
        let by_similarity = self.rank_by_similarity(word, &expanded);
        self.most_relevant(&by_similarity, prev)
    }

    fn find_similar_dictionary_entries(
        &self,
        word: &[u16],
        original: WordCase,
    ) -> Vec<Weighted<GenRoot>> {
        let d = self.speller.dictionary;
        let mut roots: JavaMaxHeap<Weighted<GenRoot>> = JavaMaxHeap::new();
        let ignore_title_case_roots = original == WordCase::Lower && !d.has_language(&["de"]);
        let trigrams = TrigramScore::new(word);
        let min = word.len().saturating_sub(MAX_ROOT_LENGTH_DIFF).max(1);
        let max = word.len() + MAX_ROOT_LENGTH_DIFF;
        d.words.process_all_words(min, max, true, |entry| {
            let root = entry.root;
            if ignore_title_case_roots
                && is_upper(root[0])
                && WordCase::case_of(root) == WordCase::Title
            {
                return;
            }
            let lower: Vec<u16> = root.iter().map(|&c| d.case_fold(c)).collect();
            let mut sc = trigrams.score(&lower);
            if sc == 0 {
                return;
            }
            sc += common_prefix(word, root) as i32 - longer_worse_penalty(word.len(), root.len());
            if roots.len() == MAX_ROOTS {
                if let Some(head) = roots.peek() {
                    if sc < head.score || sc == head.score && root > head.word.word.as_slice() {
                        return;
                    }
                }
            }
            // Lucene steps by `formStep()` over forms its flyweight entry has
            // already stripped of morphological ids: with custom morphological
            // data every other homonym is skipped. Kept.
            for form in entry.forms().step_by(d.form_step()) {
                roots.add(Weighted {
                    word: GenRoot {
                        word: root.to_vec(),
                        entry_id: form,
                    },
                    score: sc,
                });
                if roots.len() > MAX_ROOTS {
                    roots.poll();
                }
            }
        });
        let mut out = roots.queue;
        out.sort(); // stable, as `stream().sorted()`
        out
    }

    fn expand_roots(
        &self,
        misspelled: &[u16],
        roots: &[Weighted<GenRoot>],
    ) -> Vec<Weighted<Vec<u16>>> {
        let d = self.speller.dictionary;
        let thresh = calc_threshold(misspelled);
        let mut expanded: BTreeSet<Weighted<Vec<u16>>> = BTreeSet::new();
        for weighted in roots {
            for guess in self.expand_root(&weighted.word, misspelled) {
                let lower = d.to_lower_case(&guess);
                let sc = any_mismatch_ngram(misspelled.len(), misspelled, &lower, false)
                    + common_prefix(misspelled, &guess) as i32;
                if sc > thresh {
                    expanded.insert(Weighted {
                        word: guess,
                        score: sc,
                    });
                }
            }
        }
        expanded.into_iter().take(MAX_GUESSES).collect()
    }

    fn expand_root(&self, root: &GenRoot, misspelled: &[u16]) -> Vec<Vec<u16>> {
        let d = self.speller.dictionary;
        let mut cross_products: Vec<Vec<u16>> = Vec::new();
        let mut result: Vec<Vec<u16>> = Vec::new();
        let add = |result: &mut Vec<Vec<u16>>, w: Vec<u16>| {
            if !result.contains(&w) {
                result.push(w);
            }
        };
        if !d.has_flag(root.entry_id, d.needaffix) {
            add(&mut result, root.word.clone());
        }
        let word = &root.word;
        self.process_affixes(false, misspelled, &mut |suffix_len, suffix_id| {
            let strip = self.affix_strip_length(suffix_id);
            if !self.has_compatible_flags(root, suffix_id)
                || !self.check_affix_condition(
                    suffix_id,
                    word,
                    0,
                    word.len() as isize - strip as isize,
                )
            {
                return;
            }
            let suffix = &misspelled[misspelled.len() - suffix_len..];
            let with_suffix = [&word[..word.len() - strip], suffix].concat();
            add(&mut result, with_suffix.clone());
            if d.is_cross_product(suffix_id) {
                cross_products.push(with_suffix);
            }
        });
        self.process_affixes(true, misspelled, &mut |prefix_len, prefix_id| {
            if !d.has_flag(root.entry_id, d.affix_data(prefix_id, AFFIX_FLAG))
                || !d.is_cross_product(prefix_id)
            {
                return;
            }
            let strip = self.affix_strip_length(prefix_id);
            let prefix = &misspelled[..prefix_len];
            for suffixed in &cross_products {
                let stem_len = suffixed.len() as isize - strip as isize;
                if self.check_affix_condition(prefix_id, suffixed, strip, stem_len) {
                    add(&mut result, [prefix, &suffixed[strip..]].concat());
                }
            }
        });
        self.process_affixes(true, misspelled, &mut |prefix_len, prefix_id| {
            let strip = self.affix_strip_length(prefix_id);
            let stem_len = word.len() as isize - strip as isize;
            if self.has_compatible_flags(root, prefix_id)
                && self.check_affix_condition(prefix_id, word, strip, stem_len)
            {
                let prefix = &misspelled[..prefix_len];
                add(&mut result, [prefix, &word[strip..]].concat());
            }
        });
        result.truncate(MAX_WORDS);
        result
    }

    /// `processAffixes`: the affixes matching the start (prefixes) or end
    /// (suffixes) of `word`, shortest first.
    fn process_affixes(&self, prefixes: bool, word: &[u16], processor: &mut dyn FnMut(usize, i32)) {
        let d = self.speller.dictionary;
        let trie = if prefixes { &d.prefixes } else { &d.suffixes };
        let mut node = trie.root();
        if let Some(ids) = trie.finals(node) {
            for &id in ids {
                processor(0, id);
            }
        }
        let length = word.len();
        for k in 0..length {
            let i = if prefixes { k } else { length - 1 - k };
            match trie.step(node, word[i]) {
                Some(n) => node = n,
                None => break,
            }
            if let Some(ids) = trie.finals(node) {
                let affix_len = if prefixes { i + 1 } else { length - i };
                for &id in ids {
                    processor(affix_len, id);
                }
            }
        }
    }

    fn has_compatible_flags(&self, root: &GenRoot, affix_id: i32) -> bool {
        let d = self.speller.dictionary;
        if !d.has_flag(root.entry_id, d.affix_data(affix_id, AFFIX_FLAG)) {
            return false;
        }
        let append = i32::from(d.affix_data(affix_id, AFFIX_APPEND));
        !d.has_flag(append, d.needaffix)
            && !d.has_flag(append, d.circumfix)
            && !d.has_flag(append, d.onlyincompound)
    }

    fn check_affix_condition(
        &self,
        affix_id: i32,
        word: &[u16],
        offset: usize,
        length: isize,
    ) -> bool {
        if length < 0 {
            return false;
        }
        let d = self.speller.dictionary;
        let condition = d.get_affix_condition(affix_id);
        condition == 0
            || d.patterns[condition]
                .as_ref()
                .is_some_and(|c| c.accepts_stem(&word[offset..offset + length as usize]))
    }

    fn affix_strip_length(&self, affix_id: i32) -> usize {
        let d = self.speller.dictionary;
        let ord = usize::from(d.affix_data(affix_id, AFFIX_STRIP_ORD));
        d.strip_offsets[ord + 1] - d.strip_offsets[ord]
    }

    fn rank_by_similarity(
        &self,
        word: &[u16],
        expanded: &[Weighted<Vec<u16>>],
    ) -> BTreeSet<Weighted<Vec<u16>>> {
        let d = self.speller.dictionary;
        let fact = (10.0 - f64::from(d.max_diff)) / 5.0;
        let mut out = BTreeSet::new();
        for weighted in expanded {
            let guess = &weighted.word;
            let lower = d.to_lower_case(guess);
            if lower == word {
                out.insert(Weighted {
                    word: guess.clone(),
                    score: weighted.score + 2000,
                });
                break;
            }
            let re = any_mismatch_ngram(2, word, &lower, true)
                + any_mismatch_ngram(2, &lower, word, true);
            let score = 2 * lcs(word, &lower) - (word.len() as i32 - lower.len() as i32).abs()
                + common_character_position_score(word, &lower)
                + common_prefix(word, &lower) as i32
                + any_mismatch_ngram(4, word, &lower, false)
                + re
                + if f64::from(re) < (word.len() + lower.len()) as f64 * fact {
                    -1000
                } else {
                    0
                };
            out.insert(Weighted {
                word: guess.clone(),
                score,
            });
        }
        out
    }

    fn most_relevant(
        &self,
        by_similarity: &BTreeSet<Weighted<Vec<u16>>>,
        prev: &SuggestionSet,
    ) -> Vec<Vec<u16>> {
        let d = self.speller.dictionary;
        let mut result: Vec<Vec<u16>> = Vec::new();
        let mut has_excellent = false;
        for weighted in by_similarity {
            if weighted.score > 1000 {
                has_excellent = true;
            } else if has_excellent {
                break;
            }
            let bad = weighted.score < -100;
            if bad && (!result.is_empty() || d.only_max_diff) {
                break;
            }
            if !prev
                .items
                .iter()
                .any(|s| contains_sub(&weighted.word, &s.raw))
                && !result.iter().any(|r| contains_sub(&weighted.word, r))
                && self.speller.check_word(&weighted.word, None)
            {
                result.push(weighted.word.clone());
                if result.len() as i32 >= d.max_ngram_suggestions {
                    break;
                }
            }
            if bad {
                break;
            }
        }
        result
    }
}

/// `TrigramAutomaton`: scores a root by the substrings (one to three units)
/// it shares with the misspelling, each counted once, weighted by its
/// number of occurrences in the misspelling.
struct TrigramScore {
    counts: HashMap<Vec<u16>, i32>,
}

impl TrigramScore {
    fn new(s1: &[u16]) -> Self {
        let mut counts = HashMap::new();
        for start in 0..s1.len() {
            for end in start + 1..=(start + 3).min(s1.len()) {
                *counts.entry(s1[start..end].to_vec()).or_insert(0) += 1;
            }
        }
        TrigramScore { counts }
    }

    fn score(&self, s2: &[u16]) -> i32 {
        let mut seen: HashSet<&[u16]> = HashSet::new();
        let mut score = 0;
        for end in 1..=s2.len() {
            for len in 1..=3.min(end) {
                let sub = &s2[end - len..end];
                if let Some(&c) = self.counts.get(sub) {
                    if seen.insert(sub) {
                        score += c;
                    }
                }
            }
        }
        score
    }
}

/// `GeneratingSuggester.commonPrefix`.
pub(crate) fn common_prefix(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// `ngramScore`.
#[allow(clippy::needless_range_loop)] // `i` is also the substring offset, as in Java.
fn ngram_score(n: usize, s1: &[u16], s2: &[u16], weighted: bool) -> i32 {
    let l1 = s1.len();
    let mut score = 0;
    let mut last_starts = vec![0i32; l1];
    for j in 1..=n {
        let mut ns = 0;
        if j <= l1 {
            for i in 0..=(l1 - j) {
                if last_starts[i] >= 0 {
                    let pos = index_of_substring(s2, last_starts[i] as usize, s1, i, j);
                    last_starts[i] = pos;
                    if pos >= 0 {
                        ns += 1;
                        continue;
                    }
                }
                if weighted {
                    ns -= 1;
                    if i == 0 || i == l1 - j {
                        ns -= 1;
                    }
                }
            }
        }
        score += ns;
        if ns < 2 && !weighted {
            break;
        }
    }
    score
}

fn longer_worse_penalty(l1: usize, l2: usize) -> i32 {
    (l2 as i32 - l1 as i32 - 2).max(0)
}

fn any_mismatch_ngram(n: usize, s1: &[u16], s2: &[u16], weighted: bool) -> i32 {
    ngram_score(n, s1, s2, weighted) - ((s2.len() as i32 - s1.len() as i32).abs() - 2).max(0)
}

fn index_of_substring(
    hay: &[u16],
    hay_pos: usize,
    needle: &[u16],
    needle_pos: usize,
    len: usize,
) -> i32 {
    let c = needle[needle_pos];
    if hay.len() < len {
        return -1;
    }
    let limit = hay.len() - len;
    let mut i = hay_pos;
    while i <= limit {
        if hay[i] == c && hay[i + 1..i + len] == needle[needle_pos + 1..needle_pos + len] {
            return i as i32;
        }
        i += 1;
    }
    -1
}

fn lcs(s1: &[u16], s2: &[u16]) -> i32 {
    let mut lengths = vec![0i32; s2.len() + 1];
    for i in 1..=s1.len() {
        let mut prev = 0;
        for j in 1..=s2.len() {
            let cur = lengths[j];
            lengths[j] = if s1[i - 1] == s2[j - 1] {
                prev + 1
            } else {
                cur.max(lengths[j - 1])
            };
            prev = cur;
        }
    }
    lengths[s2.len()]
}

fn common_character_position_score(s1: &[u16], s2: &[u16]) -> i32 {
    let mut num = 0;
    let (mut d1, mut d2) = (0usize, 0usize);
    let mut diff = 0;
    let mut i = 0;
    while i < s1.len() && i < s2.len() {
        if s1[i] == s2[i] {
            num += 1;
        } else {
            if diff == 0 {
                d1 = i;
            } else if diff == 1 {
                d2 = i;
            }
            diff += 1;
        }
        i += 1;
    }
    let common = i32::from(num > 0);
    if diff == 2 && i == s1.len() && i == s2.len() && s1[d1] == s2[d2] && s1[d2] == s2[d1] {
        return common + 10;
    }
    common
}

/// `calcThreshold`.
fn calc_threshold(word: &[u16]) -> i32 {
    let mut thresh = 0;
    for sp in 1..4 {
        let mut mw = word.to_vec();
        let mut k = sp;
        while k < word.len() {
            mw[k] = u16::from(b'*');
            k += 4;
        }
        thresh += any_mismatch_ngram(word.len(), word, &mw, false);
    }
    thresh / 3 - 1
}
