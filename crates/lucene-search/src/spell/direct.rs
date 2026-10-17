//! `DirectSpellChecker` (M13 T13.3e): spelling suggestions read straight off
//! an index's terms, with no side index: the field's terms within one, then
//! two edits of the query term, ranked by similarity and frequency.
//!
//! Java walks a `FuzzyTermsEnum` (Levenshtein automata intersected with the
//! terms dictionary) and feeds back the queue's weakest boost through
//! `setMaxNonCompetitiveBoost`, which lowers the enum's edit budget. Here the
//! field's merged terms are walked in order and each is matched by
//! [`FuzzyMatch`]'s banded edit distance under the current budget; the
//! budget is lowered by `FuzzyTermsEnum.bottomChanged`'s rule, at the same
//! point of the walk (Java notices a new boost while returning the next
//! match and applies it before the one after), so the same terms are seen.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, OnceLock};

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::fuzzy::{FuzzyMatch, MAXIMUM_SUPPORTED_DISTANCE};

use super::distance::{LuceneLevenshteinDistance, StringDistance};
use super::suggest_word::{score_comparator, SuggestMode, SuggestWord, SuggestWordComparator};
use crate::multi_terms::MultiTerms;
use crate::reader::IndexReader;
use crate::{Error, Result};

/// `SpellChecker.DEFAULT_ACCURACY`.
pub const DEFAULT_ACCURACY: f32 = 0.5;

/// `DirectSpellChecker.INTERNAL_LEVENSHTEIN`: the distance that stands for
/// the enum's own boost. Only this instance (compared by pointer, as Java
/// compares by identity) skips the per-term distance; another
/// `LuceneLevenshteinDistance` is computed term by term.
pub fn internal_levenshtein() -> Arc<dyn StringDistance> {
    static INTERNAL: OnceLock<Arc<dyn StringDistance>> = OnceLock::new();
    INTERNAL
        .get_or_init(|| Arc::new(LuceneLevenshteinDistance::new()))
        .clone()
}

/// `String.toLowerCase(Locale.ROOT)`.
fn java_lower(s: &str) -> String {
    let units: Vec<u16> = s.encode_utf16().collect();
    String::from_utf16_lossy(&lucene_analysis::lang::java_string_to_lower_case(&units))
}

/// `Float.compare`: `-0.0 < 0.0`, and NaN equal to itself and above all.
fn float_compare(a: f32, b: f32) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        let bits = |f: f32| {
            if f.is_nan() {
                0x7fc0_0000
            } else {
                f.to_bits() as i32
            }
        };
        bits(a).cmp(&bits(b))
    }
}

/// `DirectSpellChecker.ScoreTerm`: a candidate in the walk's queue.
#[derive(Debug, Clone)]
struct ScoreTerm {
    term: Vec<u8>,
    boost: f32,
    docfreq: i32,
    term_as_string: Option<String>,
    score: f32,
}

/// `ScoreTerm.compareTo`: equal terms are equal; else by boost, ties broken
/// by the *reverse* term order. A total order over distinct terms.
impl Ord for ScoreTerm {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.term == other.term {
            return Ordering::Equal;
        }
        if self.boost == other.boost {
            other.term.cmp(&self.term)
        } else {
            float_compare(self.boost, other.boost)
        }
    }
}

impl PartialOrd for ScoreTerm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScoreTerm {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ScoreTerm {}

/// `DirectSpellChecker`.
#[derive(Clone)]
pub struct DirectSpellChecker {
    max_edits: i32,
    min_prefix: i32,
    max_inspections: i32,
    accuracy: f32,
    threshold_frequency: f32,
    min_query_length: i32,
    max_query_length: i32,
    max_query_frequency: f32,
    lower_case_terms: bool,
    comparator: SuggestWordComparator,
    distance: Arc<dyn StringDistance>,
}

impl std::fmt::Debug for DirectSpellChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectSpellChecker")
            .field("max_edits", &self.max_edits)
            .field("min_prefix", &self.min_prefix)
            .field("max_inspections", &self.max_inspections)
            .field("accuracy", &self.accuracy)
            .field("distance", &self.distance)
            .finish_non_exhaustive()
    }
}

impl Default for DirectSpellChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectSpellChecker {
    /// `new DirectSpellChecker()`: two edits, a one-character prefix, five
    /// inspections per suggestion, accuracy 0.5, queries of 4 or more
    /// characters found in at most 1% of the documents, lower-cased.
    pub fn new() -> Self {
        Self {
            max_edits: i32::from(MAXIMUM_SUPPORTED_DISTANCE),
            min_prefix: 1,
            max_inspections: 5,
            accuracy: DEFAULT_ACCURACY,
            threshold_frequency: 0.0,
            min_query_length: 4,
            max_query_length: i32::MAX,
            max_query_frequency: 0.01,
            lower_case_terms: true,
            comparator: score_comparator(),
            distance: internal_levenshtein(),
        }
    }

    /// `getMaxEdits`.
    pub fn get_max_edits(&self) -> i32 {
        self.max_edits
    }

    /// `setMaxEdits`: 1 or 2.
    ///
    /// # Errors
    /// Any other value (Java: `UnsupportedOperationException`).
    pub fn set_max_edits(&mut self, max_edits: i32) -> Result<()> {
        if max_edits < 1 || max_edits > i32::from(MAXIMUM_SUPPORTED_DISTANCE) {
            return Err(Error::Unsupported("Invalid maxEdits".into()));
        }
        self.max_edits = max_edits;
        Ok(())
    }

    /// `getMinPrefix`.
    pub fn get_min_prefix(&self) -> i32 {
        self.min_prefix
    }

    /// `setMinPrefix`: the leading characters a suggestion must share.
    pub fn set_min_prefix(&mut self, min_prefix: i32) {
        self.min_prefix = min_prefix;
    }

    /// `getMaxInspections`.
    pub fn get_max_inspections(&self) -> i32 {
        self.max_inspections
    }

    /// `setMaxInspections`: candidates gathered per requested suggestion.
    pub fn set_max_inspections(&mut self, max_inspections: i32) {
        self.max_inspections = max_inspections;
    }

    /// `getAccuracy`.
    pub fn get_accuracy(&self) -> f32 {
        self.accuracy
    }

    /// `setAccuracy`: the least similarity a suggestion may have.
    pub fn set_accuracy(&mut self, accuracy: f32) {
        self.accuracy = accuracy;
    }

    /// `getThresholdFrequency`.
    pub fn get_threshold_frequency(&self) -> f32 {
        self.threshold_frequency
    }

    /// `setThresholdFrequency`: below 1 a fraction of the documents, else
    /// an absolute document count.
    ///
    /// # Errors
    /// A fractional value of 1 or more.
    pub fn set_threshold_frequency(&mut self, threshold_frequency: f32) -> Result<()> {
        check_frequency(threshold_frequency)?;
        self.threshold_frequency = threshold_frequency;
        Ok(())
    }

    /// `getMinQueryLength`.
    pub fn get_min_query_length(&self) -> i32 {
        self.min_query_length
    }

    /// `setMinQueryLength`.
    ///
    /// # Errors
    /// Above the maximum query length.
    pub fn set_min_query_length(&mut self, min_query_length: i32) -> Result<()> {
        if min_query_length > self.max_query_length {
            return Err(Error::IllegalArgument(
                "minQueryLength must not be greater than maxQueryLength".into(),
            ));
        }
        self.min_query_length = min_query_length;
        Ok(())
    }

    /// `getMaxQueryLength`.
    pub fn get_max_query_length(&self) -> i32 {
        self.max_query_length
    }

    /// `setMaxQueryLength`.
    ///
    /// # Errors
    /// Below the minimum query length.
    pub fn set_max_query_length(&mut self, max_query_length: i32) -> Result<()> {
        if max_query_length < self.min_query_length {
            return Err(Error::IllegalArgument(
                "maxQueryLength must not be smaller than minQueryLength".into(),
            ));
        }
        self.max_query_length = max_query_length;
        Ok(())
    }

    /// `getMaxQueryFrequency`.
    pub fn get_max_query_frequency(&self) -> f32 {
        self.max_query_frequency
    }

    /// `setMaxQueryFrequency`: a query term in more documents than this
    /// (a fraction below 1, else a count) is taken as correctly spelled.
    ///
    /// # Errors
    /// A fractional value of 1 or more.
    pub fn set_max_query_frequency(&mut self, max_query_frequency: f32) -> Result<()> {
        check_frequency(max_query_frequency)?;
        self.max_query_frequency = max_query_frequency;
        Ok(())
    }

    /// `getLowerCaseTerms`.
    pub fn get_lower_case_terms(&self) -> bool {
        self.lower_case_terms
    }

    /// `setLowerCaseTerms`: lower-case the query term (`Locale.ROOT`).
    pub fn set_lower_case_terms(&mut self, lower_case_terms: bool) {
        self.lower_case_terms = lower_case_terms;
    }

    /// `getComparator`.
    pub fn get_comparator(&self) -> SuggestWordComparator {
        self.comparator.clone()
    }

    /// `setComparator`: the suggestions' order, the greatest first.
    pub fn set_comparator(&mut self, comparator: SuggestWordComparator) {
        self.comparator = comparator;
    }

    /// `getDistance`.
    pub fn get_distance(&self) -> Arc<dyn StringDistance> {
        self.distance.clone()
    }

    /// `setDistance`: the similarity a suggestion is scored by;
    /// [`internal_levenshtein`] scores by the walk's own edit distance.
    pub fn set_distance(&mut self, distance: Arc<dyn StringDistance>) {
        self.distance = distance;
    }

    /// `suggestSimilar(term, numSug, ir)`: `SUGGEST_WHEN_NOT_IN_INDEX`.
    ///
    /// # Errors
    /// Reading the index fails.
    pub fn suggest_similar<R: IndexReader + ?Sized>(
        &self,
        field: &str,
        text: &str,
        num_sug: usize,
        ir: &R,
    ) -> Result<Vec<SuggestWord>> {
        self.suggest_similar_with_mode(field, text, num_sug, ir, SuggestMode::SuggestWhenNotInIndex)
    }

    /// `suggestSimilar(term, numSug, ir, suggestMode)`: the configured
    /// accuracy.
    ///
    /// # Errors
    /// Reading the index fails.
    pub fn suggest_similar_with_mode<R: IndexReader + ?Sized>(
        &self,
        field: &str,
        text: &str,
        num_sug: usize,
        ir: &R,
        suggest_mode: SuggestMode,
    ) -> Result<Vec<SuggestWord>> {
        self.suggest_similar_with_accuracy(field, text, num_sug, ir, suggest_mode, self.accuracy)
    }

    /// `suggestSimilar(term, numSug, ir, suggestMode, accuracy)`: at most
    /// `num_sug` suggestions for `text` in `field`, the best first.
    ///
    /// Departure from Java: when `num_sug * maxInspections` is not positive
    /// the walk keeps nothing (Java throws a `NullPointerException` on the
    /// first candidate).
    ///
    /// # Errors
    /// Reading the index fails.
    pub fn suggest_similar_with_accuracy<R: IndexReader + ?Sized>(
        &self,
        field: &str,
        text: &str,
        num_sug: usize,
        ir: &R,
        suggest_mode: SuggestMode,
        accuracy: f32,
    ) -> Result<Vec<SuggestWord>> {
        let text_length = text.chars().count();
        if (text_length as i64) < i64::from(self.min_query_length)
            || (text_length as i64) > i64::from(self.max_query_length)
        {
            return Ok(Vec::new());
        }
        let text = if self.lower_case_terms {
            java_lower(text)
        } else {
            text.to_string()
        };
        // Java's `IndexReader.docFreq` is an int
        let mut docfreq = ir.doc_freq(field, text.as_bytes())? as i32;
        if suggest_mode == SuggestMode::SuggestWhenNotInIndex && docfreq > 0 {
            return Ok(Vec::new());
        }
        let max_doc = ir.max_doc();
        // Java's `if`/`else if`, both answering nothing
        if (self.max_query_frequency >= 1.0 && docfreq as f32 > self.max_query_frequency)
            || docfreq > f64::from(self.max_query_frequency * max_doc as f32).ceil() as i32
        {
            return Ok(Vec::new());
        }
        if suggest_mode != SuggestMode::SuggestMorePopular {
            docfreq = 0;
        }
        if self.threshold_frequency >= 1.0 {
            docfreq = docfreq.max(self.threshold_frequency as i32);
        } else if self.threshold_frequency > 0.0 {
            docfreq =
                docfreq.max(((self.threshold_frequency * max_doc as f32) as i32).wrapping_sub(1));
        }

        let inspections = (num_sug as i64).saturating_mul(i64::from(self.max_inspections));
        // try ed=1 first, in case we get lucky
        let mut terms = self.walk(field, &text, inspections, ir, docfreq, 1, accuracy)?;
        if self.max_edits > 1 && (terms.len() as i64) < inspections {
            let more = self.walk(
                field,
                &text,
                inspections,
                ir,
                docfreq,
                self.max_edits,
                accuracy,
            )?;
            // a HashSet: the first pass's entry wins
            let mut seen: HashMap<Vec<u8>, ()> =
                terms.iter().map(|t| (t.term.clone(), ())).collect();
            for t in more {
                if seen.insert(t.term.clone(), ()).is_none() {
                    terms.push(t);
                }
            }
        }

        let suggestions: Vec<SuggestWord> = terms
            .into_iter()
            .map(|s| SuggestWord {
                string: s
                    .term_as_string
                    .unwrap_or_else(|| String::from_utf8_lossy(&s.term).into_owned()),
                score: s.score,
                freq: s.docfreq,
            })
            .collect();
        // `timSort(reverseOrder(comparator))`: stable, the greatest first
        let order = java_sort_order(&suggestions, |a, b| (self.comparator)(b, a));
        let mut slots: Vec<Option<SuggestWord>> = suggestions.into_iter().map(Some).collect();
        Ok(order
            .into_iter()
            .take(num_sug)
            .filter_map(|i| slots[i].take())
            .collect())
    }

    /// `suggestSimilar(term, numSug, ir, docfreq, editDistance, accuracy,
    /// spare)`: the walk, keeping the `num_sug` best candidates.
    #[allow(clippy::too_many_arguments)]
    fn walk<R: IndexReader + ?Sized>(
        &self,
        field: &str,
        text: &str,
        num_sug: i64,
        ir: &R,
        docfreq: i32,
        edit_distance: i32,
        accuracy: f32,
    ) -> Result<Vec<ScoreTerm>> {
        let Some(terms) = MultiTerms::get_terms(ir, field)? else {
            return Ok(Vec::new());
        };
        if num_sug <= 0 {
            return Ok(Vec::new());
        }
        let num_sug = num_sug as usize;
        let prefix = self.min_prefix.max(edit_distance - 1).max(0) as usize;
        let query = text.as_bytes();
        let pattern = FuzzyMatch::new(query, edit_distance as u8, prefix, true);
        let term_length = pattern.term_length();
        let internal = Arc::ptr_eq(&self.distance, &internal_levenshtein());

        // FuzzyTermsEnum's state: its edit budget, the boost it last saw,
        // whether a change waits for the next call, and the attribute
        let mut max_edits = edit_distance;
        let mut bottom = f32::NEG_INFINITY;
        let mut queued = false;
        let mut max_non_competitive = f32::NEG_INFINITY;
        let bottom_changed = |max_edits: &mut i32, bottom: f32| {
            while *max_edits > 0 {
                let max_boost = 1.0 - (*max_edits as f32 / term_length as f32);
                if bottom < max_boost {
                    break;
                }
                *max_edits -= 1;
            }
        };
        bottom_changed(&mut max_edits, bottom);

        // java.util.PriorityQueue: the least on top
        let mut st_queue: BinaryHeap<std::cmp::Reverse<ScoreTerm>> = BinaryHeap::new();
        let peek_boost = |q: &BinaryHeap<std::cmp::Reverse<ScoreTerm>>| {
            q.peek().map_or(f32::NEG_INFINITY, |r| r.0.boost)
        };
        // Every match starts with the fixed prefix, so the walk covers only
        // the prefix's range of the terms dictionary, as the automaton does
        let literal = pattern.literal_prefix();
        let mut te = terms.iterator()?;
        let mut first = true;
        loop {
            if queued {
                bottom_changed(&mut max_edits, bottom);
                queued = false;
            }
            let candidate = if std::mem::take(&mut first) && !literal.is_empty() {
                match te.try_seek_ceil(literal)? {
                    SeekStatus::End => None,
                    SeekStatus::Found | SeekStatus::NotFound => te.term(),
                }
            } else {
                te.next()?
            };
            let Some(candidate) = candidate.filter(|c| c.starts_with(literal)) else {
                break;
            };
            let Some(ed) = pattern.edits_within(candidate, max_edits as u8) else {
                continue;
            };
            let candidate = candidate.to_vec();
            let mut score = pattern.boost_from_edits(&candidate, ed);
            // Java's float `!=`: a NaN always differs, -0.0 equals 0.0
            if max_non_competitive != bottom {
                bottom = max_non_competitive;
                queued = true;
            }

            // ignore uncompetitive hits
            if st_queue.len() >= num_sug && score <= peek_boost(&st_queue) {
                continue;
            }
            // ignore exact match of the same term
            if candidate == query {
                continue;
            }
            let df = te.doc_freq()?;
            // check docFreq if required
            if df <= docfreq {
                continue;
            }
            let term_as_string = if internal {
                // delay creating strings until the end
                None
            } else {
                let s = String::from_utf8_lossy(&candidate).into_owned();
                score = self.distance.get_distance(text, &s);
                Some(s)
            };
            if score < accuracy {
                continue;
            }
            st_queue.push(std::cmp::Reverse(ScoreTerm {
                term: candidate,
                boost: score,
                docfreq: df,
                term_as_string,
                score,
            }));
            // possibly drop entries from queue
            if st_queue.len() > num_sug {
                st_queue.pop();
            }
            max_non_competitive = if st_queue.len() >= num_sug {
                peek_boost(&st_queue)
            } else {
                f32::NEG_INFINITY
            };
        }
        Ok(st_queue.into_iter().map(|r| r.0).collect())
    }
}

/// `setThresholdFrequency`/`setMaxQueryFrequency`'s check.
fn check_frequency(value: f32) -> Result<()> {
    if value >= 1.0 && value != (value as i32) as f32 {
        return Err(Error::IllegalArgument(
            "Fractional absolute document frequencies are not allowed".into(),
        ));
    }
    Ok(())
}

/// The order `TimSort` leaves `items` in, as indexes. The score comparator
/// lets a NaN score tie with everything, so the order is not total and
/// `slice::sort_by` may panic on it. Below `TimSort.MIN_MERGE` (32) this is
/// Java's own path -- the leading run made ascending, then a binary
/// insertion sort -- so even such an order comes out as Java's; above it, a
/// stable merge sort, which equals `TimSort` whenever the order is total.
fn java_sort_order<T>(items: &[T], cmp: impl Fn(&T, &T) -> Ordering) -> Vec<usize> {
    let mut a: Vec<usize> = (0..items.len()).collect();
    let less = |x: usize, y: usize| cmp(&items[x], &items[y]) == Ordering::Less;
    if a.len() >= 32 {
        merge_sort(&mut a, &less);
        return a;
    }
    if a.len() < 2 {
        return a;
    }
    // countRunAndMakeAscending
    let mut run = 2;
    if less(a[1], a[0]) {
        while run < a.len() && less(a[run], a[run - 1]) {
            run += 1;
        }
        a[..run].reverse();
    } else {
        while run < a.len() && !less(a[run], a[run - 1]) {
            run += 1;
        }
    }
    // binarySort
    for start in run..a.len() {
        let pivot = a[start];
        let (mut left, mut right) = (0, start);
        while left < right {
            let mid = (left + right) / 2;
            if less(pivot, a[mid]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        a.copy_within(left..start, left + 1);
        a[left] = pivot;
    }
    a
}

/// A stable top-down merge sort that never panics on an inconsistent order.
fn merge_sort(a: &mut [usize], less: &impl Fn(usize, usize) -> bool) {
    if a.len() < 2 {
        return;
    }
    let mid = a.len() / 2;
    merge_sort(&mut a[..mid], less);
    merge_sort(&mut a[mid..], less);
    let (left, right) = (a[..mid].to_vec(), a[mid..].to_vec());
    let (mut i, mut j) = (0, 0);
    for slot in a.iter_mut() {
        // the left run wins ties: stable
        if j >= right.len() || (i < left.len() && !less(right[j], left[i])) {
            *slot = left[i];
            i += 1;
        } else {
            *slot = right[j];
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sort_is_timsorts_and_never_panics() {
        // NaN ties with everything, so to TimSort this is one ascending run
        // and nothing moves
        let v = [1.0f32, f32::NAN, 0.5, 2.0, f32::NAN, 0.25];
        let by = |a: &f32, b: &f32| {
            if a > b {
                Ordering::Greater
            } else if a < b {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        };
        assert_eq!(java_sort_order(&v, by), [0, 1, 2, 3, 4, 5]);
        // a descending run is reversed, then the rest inserted
        assert_eq!(java_sort_order(&[3.0, 2.0, 1.0, 2.5], by), [2, 1, 3, 0]);
        // a long list, merged stably
        let long: Vec<i32> = (0..100).map(|i| (i * 37) % 10).collect();
        let order = java_sort_order(&long, |a, b| a.cmp(b));
        assert!(order
            .windows(2)
            .all(|w| (long[w[0]], w[0]) < (long[w[1]], w[1])));
        let nan: Vec<f32> = (0..50)
            .map(|i| if i % 3 == 0 { f32::NAN } else { i as f32 })
            .collect();
        assert_eq!(java_sort_order(&nan, by).len(), 50);
        assert!(java_sort_order::<f32>(&[], by).is_empty());
    }

    #[test]
    fn float_compare_is_javas() {
        assert_eq!(float_compare(-0.0, 0.0), Ordering::Less);
        assert_eq!(float_compare(f32::NAN, f32::NAN), Ordering::Equal);
        assert_eq!(float_compare(f32::NAN, 1.0), Ordering::Greater);
        assert_eq!(float_compare(1.0, 2.0), Ordering::Less);
        assert_eq!(float_compare(2.0, 1.0), Ordering::Greater);
    }

    #[test]
    fn score_terms_order_by_boost_then_reverse_term() {
        let st = |t: &str, b: f32| ScoreTerm {
            term: t.as_bytes().to_vec(),
            boost: b,
            docfreq: 1,
            term_as_string: None,
            score: b,
        };
        assert_eq!(st("a", 0.5).cmp(&st("a", 0.9)), Ordering::Equal);
        assert_eq!(st("a", 0.5).cmp(&st("b", 0.5)), Ordering::Greater);
        assert_eq!(st("a", 0.5).cmp(&st("b", 0.6)), Ordering::Less);
        assert!(st("a", 0.5) == st("a", 0.1));
        assert_eq!(
            st("a", 0.5).partial_cmp(&st("b", 0.5)),
            Some(Ordering::Greater)
        );
    }
}
