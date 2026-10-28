//! `WordBreakSpellChecker` (M13 T13.3e): suggestions made by breaking one
//! query term into several indexed words, or by joining adjacent query terms
//! into one.

use std::cmp::Ordering;

use super::suggest_word::{CombineSuggestion, SuggestMode, SuggestWord};
use crate::query_visitor::Term;
use crate::reader::IndexReader;
use crate::Result;

/// `java.util.PriorityQueue` with a comparator: `offer`, `poll` and `size`
/// as `siftUpUsingComparator`/`siftDownUsingComparator` do them, so elements
/// that tie leave in Java's order.
struct JavaPriorityQueue<T> {
    es: Vec<T>,
    cmp: fn(&T, &T) -> Ordering,
}

impl<T> JavaPriorityQueue<T> {
    fn new(cmp: fn(&T, &T) -> Ordering) -> Self {
        Self {
            es: Vec::new(),
            cmp,
        }
    }

    fn len(&self) -> usize {
        self.es.len()
    }

    fn offer(&mut self, x: T) {
        let mut k = self.es.len();
        self.es.push(x);
        while k > 0 {
            let parent = (k - 1) >> 1;
            if (self.cmp)(&self.es[k], &self.es[parent]).is_ge() {
                break;
            }
            self.es.swap(k, parent);
            k = parent;
        }
    }

    fn poll(&mut self) -> Option<T> {
        if self.es.is_empty() {
            return None;
        }
        let last = self.es.len() - 1;
        self.es.swap(0, last);
        let result = self.es.pop();
        let n = self.es.len();
        let mut k = 0;
        let half = n >> 1;
        while k < half {
            let mut child = (k << 1) + 1;
            let right = child + 1;
            if right < n && (self.cmp)(&self.es[child], &self.es[right]).is_gt() {
                child = right;
            }
            if (self.cmp)(&self.es[k], &self.es[child]).is_le() {
                break;
            }
            self.es.swap(k, child);
            k = child;
        }
        result
    }
}

/// A Java `int` comparison result, `a - b` with wrap-around, as an ordering.
fn int_diff(a: i32, b: i32) -> Ordering {
    a.wrapping_sub(b).cmp(&0)
}

/// `WordBreakSpellChecker.BreakSuggestionSortMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BreakSuggestionSortMethod {
    /// `NUM_CHANGES_THEN_SUMMED_FREQUENCY`.
    NumChangesThenSummedFrequency,
    /// `NUM_CHANGES_THEN_MAX_FREQUENCY`.
    NumChangesThenMaxFrequency,
}

/// `SuggestWordArrayWrapper`: a break suggestion and its frequencies.
struct SuggestWordArrayWrapper {
    suggest_words: Vec<SuggestWord>,
    freq_max: i32,
    freq_sum: i32,
}

impl SuggestWordArrayWrapper {
    fn new(suggest_words: Vec<SuggestWord>) -> Self {
        let mut freq_sum = 0i32;
        let mut freq_max = 0i32;
        for sw in &suggest_words {
            freq_sum = freq_sum.wrapping_add(sw.freq);
            freq_max = freq_max.max(sw.freq);
        }
        Self {
            suggest_words,
            freq_max,
            freq_sum,
        }
    }

    fn length(&self) -> i32 {
        self.suggest_words.len() as i32
    }
}

/// `LengthThenMaxFreqComparator`: fewer words is greater, then the higher
/// maximum frequency.
fn length_then_max_freq(o1: &SuggestWordArrayWrapper, o2: &SuggestWordArrayWrapper) -> Ordering {
    if o1.length() != o2.length() {
        return int_diff(o2.length(), o1.length());
    }
    if o1.freq_max != o2.freq_max {
        return int_diff(o1.freq_max, o2.freq_max);
    }
    Ordering::Equal
}

/// `LengthThenSumFreqComparator`: fewer words is greater, then the higher
/// summed frequency.
fn length_then_sum_freq(o1: &SuggestWordArrayWrapper, o2: &SuggestWordArrayWrapper) -> Ordering {
    if o1.length() != o2.length() {
        return int_diff(o2.length(), o1.length());
    }
    if o1.freq_sum != o2.freq_sum {
        return int_diff(o1.freq_sum, o2.freq_sum);
    }
    Ordering::Equal
}

/// `CombineSuggestionWrapper`.
struct CombineSuggestionWrapper {
    combine_suggestion: CombineSuggestion,
    num_combinations: i32,
}

/// `CombinationsThenFreqComparator`: fewer joined terms is greater, then the
/// higher frequency.
fn combinations_then_freq(
    o1: &CombineSuggestionWrapper,
    o2: &CombineSuggestionWrapper,
) -> Ordering {
    if o1.num_combinations != o2.num_combinations {
        return int_diff(o2.num_combinations, o1.num_combinations);
    }
    let (f1, f2) = (
        o1.combine_suggestion.suggestion.freq,
        o2.combine_suggestion.suggestion.freq,
    );
    if f1 != f2 {
        return int_diff(f1, f2);
    }
    Ordering::Equal
}

/// `WordBreakSpellChecker`.
#[derive(Debug, Clone)]
pub struct WordBreakSpellChecker {
    min_suggestion_frequency: i32,
    min_break_word_length: i32,
    max_combine_word_length: i32,
    max_changes: i32,
    max_evaluations: i32,
}

impl Default for WordBreakSpellChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// `SEPARATOR_TERM`: `new Term("", "")`, which no suggestion joins across.
pub fn separator_term() -> Term {
    Term::new("", Vec::new())
}

fn text_of(term: &Term) -> String {
    String::from_utf8_lossy(&term.bytes).into_owned()
}

/// Java `IndexReader.docFreq` is an `int`.
fn doc_freq<R: IndexReader + ?Sized>(ir: &R, field: &str, text: &str) -> Result<i32> {
    Ok(ir.doc_freq(field, text.as_bytes())? as i32)
}

impl WordBreakSpellChecker {
    /// `new WordBreakSpellChecker()`: minimum frequency 1, words of at least
    /// one character, joins up to 20 characters, one change, 1000
    /// evaluations.
    pub fn new() -> Self {
        Self {
            min_suggestion_frequency: 1,
            min_break_word_length: 1,
            max_combine_word_length: 20,
            max_changes: 1,
            max_evaluations: 1000,
        }
    }

    /// `suggestWordBreaks`: up to `max_suggestions` ways to split `term` into
    /// indexed words, the best first. `None` mode or sort method takes
    /// Java's default for a `null` (`SUGGEST_WHEN_NOT_IN_INDEX`,
    /// `NUM_CHANGES_THEN_MAX_FREQUENCY`).
    ///
    /// # Errors
    /// Reading the index fails.
    pub fn suggest_word_breaks<R: IndexReader + ?Sized>(
        &self,
        term: &Term,
        max_suggestions: i32,
        ir: &R,
        suggest_mode: Option<SuggestMode>,
        sort_method: Option<BreakSuggestionSortMethod>,
    ) -> Result<Vec<Vec<SuggestWord>>> {
        if max_suggestions < 1 {
            return Ok(Vec::new());
        }
        let suggest_mode = suggest_mode.unwrap_or(SuggestMode::SuggestWhenNotInIndex);
        let sort_method =
            sort_method.unwrap_or(BreakSuggestionSortMethod::NumChangesThenMaxFrequency);
        let mut suggestions = JavaPriorityQueue::new(match sort_method {
            BreakSuggestionSortMethod::NumChangesThenMaxFrequency => length_then_max_freq,
            BreakSuggestionSortMethod::NumChangesThenSummedFrequency => length_then_sum_freq,
        });
        let orig_freq = doc_freq(ir, &term.field, &text_of(term))?;
        if orig_freq > 0 && suggest_mode == SuggestMode::SuggestWhenNotInIndex {
            return Ok(Vec::new());
        }
        let mut use_min_suggestion_frequency = self.min_suggestion_frequency;
        if suggest_mode == SuggestMode::SuggestMorePopular {
            use_min_suggestion_frequency = if orig_freq == 0 { 1 } else { orig_freq };
        }
        self.generate_break_up_suggestions(
            &term.field,
            &text_of(term),
            ir,
            1,
            max_suggestions,
            use_min_suggestion_frequency,
            &[],
            &mut suggestions,
            0,
        )?;
        let mut out = Vec::with_capacity(suggestions.len());
        while let Some(s) = suggestions.poll() {
            out.push(s.suggest_words);
        }
        out.reverse();
        Ok(out)
    }

    /// `suggestWordCombinations`: up to `max_suggestions` words made by
    /// joining runs of adjacent `terms` (never across [`separator_term`]),
    /// the best first. The joined term takes the first term's field.
    ///
    /// # Errors
    /// Reading the index fails.
    pub fn suggest_word_combinations<R: IndexReader + ?Sized>(
        &self,
        terms: &[Term],
        max_suggestions: i32,
        ir: &R,
        suggest_mode: SuggestMode,
    ) -> Result<Vec<CombineSuggestion>> {
        if max_suggestions < 1 {
            return Ok(Vec::new());
        }
        let mut orig_freqs = None;
        if suggest_mode != SuggestMode::SuggestAlways {
            let mut freqs = Vec::with_capacity(terms.len());
            for t in terms {
                freqs.push(doc_freq(ir, &t.field, &text_of(t))?);
            }
            orig_freqs = Some(freqs);
        }
        let mut suggestions = JavaPriorityQueue::new(combinations_then_freq);
        let separator = separator_term();
        let mut this_time_evaluations = 0i32;
        for i in 0..terms.len().saturating_sub(1) {
            if terms[i] == separator {
                continue;
            }
            let left_term_text = text_of(&terms[i]);
            let left_term_length = left_term_text.chars().count() as i64;
            if left_term_length > i64::from(self.max_combine_word_length) {
                continue;
            }
            let (mut max_freq, mut min_freq) = match &orig_freqs {
                Some(f) => (f[i], f[i]),
                None => (0, i32::MAX),
            };
            let mut combined_term_text = left_term_text;
            let mut combined_length = left_term_length;
            let mut j = i + 1;
            while j < terms.len() && ((j - i) as i64) <= i64::from(self.max_changes) {
                if terms[j] == separator {
                    break;
                }
                let right_term_text = text_of(&terms[j]);
                combined_length += right_term_text.chars().count() as i64;
                combined_term_text.push_str(&right_term_text);
                if combined_length > i64::from(self.max_combine_word_length) {
                    break;
                }
                if let Some(f) = &orig_freqs {
                    max_freq = max_freq.max(f[j]);
                    min_freq = min_freq.min(f[j]);
                }
                let combined_term_freq = doc_freq(ir, &terms[0].field, &combined_term_text)?;
                if (suggest_mode != SuggestMode::SuggestMorePopular
                    || combined_term_freq >= max_freq)
                    && (suggest_mode != SuggestMode::SuggestWhenNotInIndex || min_freq == 0)
                    && combined_term_freq >= self.min_suggestion_frequency
                {
                    let orig_indexes: Vec<i32> = (i..=j).map(|k| k as i32).collect();
                    let num_combinations = orig_indexes.len() as i32 - 1;
                    let word = SuggestWord {
                        freq: combined_term_freq,
                        score: num_combinations as f32,
                        string: combined_term_text.clone(),
                    };
                    suggestions.offer(CombineSuggestionWrapper {
                        combine_suggestion: CombineSuggestion {
                            suggestion: word,
                            original_term_indexes: orig_indexes,
                        },
                        num_combinations,
                    });
                    if suggestions.len() as i64 > i64::from(max_suggestions) {
                        suggestions.poll();
                    }
                }
                this_time_evaluations = this_time_evaluations.wrapping_add(1);
                if this_time_evaluations == self.max_evaluations {
                    break;
                }
                j += 1;
            }
        }
        let mut out = Vec::with_capacity(suggestions.len());
        while let Some(s) = suggestions.poll() {
            out.push(s.combine_suggestion);
        }
        out.reverse();
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_break_up_suggestions<R: IndexReader + ?Sized>(
        &self,
        field: &str,
        term_text: &str,
        ir: &R,
        number_breaks: i32,
        max_suggestions: i32,
        use_min_suggestion_frequency: i32,
        prefix: &[SuggestWord],
        suggestions: &mut JavaPriorityQueue<SuggestWordArrayWrapper>,
        mut total_evaluations: i32,
    ) -> Result<i32> {
        // the byte offset of each code point, and the end
        let boundaries: Vec<usize> = term_text
            .char_indices()
            .map(|(b, _)| b)
            .chain(std::iter::once(term_text.len()))
            .collect();
        let term_length = (boundaries.len() - 1) as i64;
        let use_min_break_word_length = i64::from(self.min_break_word_length.max(1));
        if term_length < use_min_break_word_length * 2 {
            return Ok(total_evaluations);
        }
        let max_split_position = term_length - use_min_break_word_length;
        // Java's FixedBitSet over UTF-16 offsets, walked in order: the byte
        // offsets of the same split points, in the same order, each with its
        // left word (Java looks it up again in the second phase; its
        // frequency cannot have changed)
        let mut valid_left_split_positions = Vec::new();
        for i in use_min_break_word_length..=max_split_position {
            if total_evaluations >= self.max_evaluations {
                return Ok(total_evaluations);
            }
            total_evaluations = total_evaluations.wrapping_add(1);
            let end = boundaries[i as usize];
            let (left_text, right_text) = term_text.split_at(end);
            let left_word = Self::generate_suggest_word(ir, field, left_text)?;
            if left_word.freq >= use_min_suggestion_frequency {
                valid_left_split_positions.push((end, left_word.clone()));
                let right_word = Self::generate_suggest_word(ir, field, right_text)?;
                if right_word.freq >= use_min_suggestion_frequency {
                    suggestions.offer(SuggestWordArrayWrapper::new(Self::new_suggestion(
                        prefix, left_word, right_word,
                    )));
                    if suggestions.len() as i64 > i64::from(max_suggestions) {
                        suggestions.poll();
                    }
                }
            }
        }
        let new_number_breaks = number_breaks.wrapping_add(1);
        if total_evaluations >= self.max_evaluations
            || new_number_breaks > self.max_changes
            || suggestions.len() as i64 >= i64::from(max_suggestions)
        {
            return Ok(total_evaluations);
        }
        for (pos, left_word) in valid_left_split_positions {
            let right_text = &term_text[pos..];
            let mut new_prefix = prefix.to_vec();
            new_prefix.push(left_word);
            total_evaluations = self.generate_break_up_suggestions(
                field,
                right_text,
                ir,
                new_number_breaks,
                max_suggestions,
                use_min_suggestion_frequency,
                &new_prefix,
                suggestions,
                total_evaluations,
            )?;
            if total_evaluations >= self.max_evaluations {
                break;
            }
        }
        Ok(total_evaluations)
    }

    /// `newSuggestion`: the prefix's words then the two new ones, each
    /// scored by the number of breaks.
    fn new_suggestion(
        prefix: &[SuggestWord],
        mut append1: SuggestWord,
        mut append2: SuggestWord,
    ) -> Vec<SuggestWord> {
        let score = (prefix.len() + 1) as f32;
        let mut out: Vec<SuggestWord> = prefix
            .iter()
            .map(|p| SuggestWord {
                string: p.string.clone(),
                freq: p.freq,
                score,
            })
            .collect();
        append1.score = score;
        append2.score = score;
        out.push(append1);
        out.push(append2);
        out
    }

    /// `generateSuggestWord`: `text` with its document frequency, score 1.
    fn generate_suggest_word<R: IndexReader + ?Sized>(
        ir: &R,
        field: &str,
        text: &str,
    ) -> Result<SuggestWord> {
        Ok(SuggestWord {
            freq: doc_freq(ir, field, text)?,
            score: 1.0,
            string: text.to_string(),
        })
    }

    /// `getMinSuggestionFrequency`.
    pub fn get_min_suggestion_frequency(&self) -> i32 {
        self.min_suggestion_frequency
    }

    /// `getMaxCombineWordLength`.
    pub fn get_max_combine_word_length(&self) -> i32 {
        self.max_combine_word_length
    }

    /// `getMinBreakWordLength`.
    pub fn get_min_break_word_length(&self) -> i32 {
        self.min_break_word_length
    }

    /// `getMaxChanges`.
    pub fn get_max_changes(&self) -> i32 {
        self.max_changes
    }

    /// `getMaxEvaluations`.
    pub fn get_max_evaluations(&self) -> i32 {
        self.max_evaluations
    }

    /// `setMinSuggestionFrequency`: the least document frequency a
    /// suggested word may have.
    pub fn set_min_suggestion_frequency(&mut self, v: i32) {
        self.min_suggestion_frequency = v;
    }

    /// `setMaxCombineWordLength`: the longest joined word, in characters.
    pub fn set_max_combine_word_length(&mut self, v: i32) {
        self.max_combine_word_length = v;
    }

    /// `setMinBreakWordLength`: the shortest word a break may make.
    pub fn set_min_break_word_length(&mut self, v: i32) {
        self.min_break_word_length = v;
    }

    /// `setMaxChanges`: the most breaks, or terms joined, per suggestion.
    pub fn set_max_changes(&mut self, v: i32) {
        self.max_changes = v;
    }

    /// `setMaxEvaluations`: the most candidate words looked up.
    pub fn set_max_evaluations(&mut self, v: i32) {
        self.max_evaluations = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_java_queue_polls_ties_in_heap_order() {
        let mut q: JavaPriorityQueue<(i32, char)> = JavaPriorityQueue::new(|a, b| a.0.cmp(&b.0));
        for (i, c) in [(2, 'a'), (1, 'b'), (2, 'c'), (1, 'd'), (0, 'e'), (2, 'f')] {
            q.offer((i, c));
        }
        let mut out = Vec::new();
        while let Some(x) = q.poll() {
            out.push(x);
        }
        // java.util.PriorityQueue's heap gives e, b, d, a, c, f for these
        assert_eq!(
            out,
            [(0, 'e'), (1, 'b'), (1, 'd'), (2, 'a'), (2, 'c'), (2, 'f')]
        );
        assert!(q.poll().is_none());
    }

    #[test]
    fn java_int_comparisons_wrap() {
        assert_eq!(int_diff(i32::MIN, 1), Ordering::Greater);
        assert_eq!(int_diff(1, 2), Ordering::Less);
    }
}
