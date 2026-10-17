//! `SuggestWord`, its two comparators, `SuggestWordQueue`, `SuggestMode` and
//! `CombineSuggestion` (M13 T13.3e): the values the spell checkers answer
//! with and the order they rank them in.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use crate::function::java_float;
use crate::suggest::lookup::charsequence_compare;
use crate::{Error, Result};

/// `SuggestWord`: a suggested term, its score and its document frequency.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SuggestWord {
    /// `score`: the similarity to the misspelled word.
    pub score: f32,
    /// `freq`: the term's document frequency.
    pub freq: i32,
    /// `string`: the suggested term.
    pub string: String,
}

impl SuggestWord {
    /// `new SuggestWord()` with its fields set.
    pub fn new(string: impl Into<String>, score: f32, freq: i32) -> Self {
        Self {
            score,
            freq,
            string: string.into(),
        }
    }
}

impl fmt::Display for SuggestWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SuggestWord(string={}, score={}, freq={})",
            self.string,
            java_float(self.score),
            self.freq
        )
    }
}

/// A `Comparator<SuggestWord>`: the spell checkers rank by it, the greatest
/// first.
pub type SuggestWordComparator = Arc<dyn Fn(&SuggestWord, &SuggestWord) -> Ordering + Send + Sync>;

/// `first > second` is greater, `first < second` less; neither (equal, or a
/// NaN score) falls through to the next key, as Java's `if` chains do.
fn by<T: PartialOrd>(first: T, second: T) -> Option<Ordering> {
    if first > second {
        Some(Ordering::Greater)
    } else if first < second {
        Some(Ordering::Less)
    } else {
        None
    }
}

/// `SuggestWordScoreComparator.compare`: score, then frequency, then the
/// *reverse* of `String.compareTo`, so the alphabetically first word ranks
/// highest.
pub fn compare_by_score(first: &SuggestWord, second: &SuggestWord) -> Ordering {
    by(first.score, second.score)
        .or_else(|| by(first.freq, second.freq))
        .unwrap_or_else(|| charsequence_compare(&second.string, &first.string))
}

/// `SuggestWordFrequencyComparator.compare`: frequency, then score, then
/// the reverse of `String.compareTo`.
pub fn compare_by_frequency(first: &SuggestWord, second: &SuggestWord) -> Ordering {
    by(first.freq, second.freq)
        .or_else(|| by(first.score, second.score))
        .unwrap_or_else(|| charsequence_compare(&second.string, &first.string))
}

/// `new SuggestWordScoreComparator()`.
pub fn score_comparator() -> SuggestWordComparator {
    Arc::new(compare_by_score)
}

/// `new SuggestWordFrequencyComparator()`.
pub fn frequency_comparator() -> SuggestWordComparator {
    Arc::new(compare_by_frequency)
}

/// `SuggestWordQueue`: Lucene's bounded `PriorityQueue` of [`SuggestWord`]s,
/// the least by the comparator on top, heap operation for heap operation, so
/// words that tie come out in Lucene's order.
pub struct SuggestWordQueue {
    /// `heap[1..=size]` in Java's 1-based layout, stored from index 0.
    heap: Vec<SuggestWord>,
    max_size: usize,
    comparator: SuggestWordComparator,
}

impl fmt::Debug for SuggestWordQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SuggestWordQueue")
            .field("heap", &self.heap)
            .field("max_size", &self.max_size)
            .finish_non_exhaustive()
    }
}

impl SuggestWordQueue {
    /// `new SuggestWordQueue(size)`: `DEFAULT_COMPARATOR`, the score
    /// comparator.
    pub fn new(size: usize) -> Self {
        Self::with_comparator(size, score_comparator())
    }

    /// `new SuggestWordQueue(size, comparator)`.
    pub fn with_comparator(size: usize, comparator: SuggestWordComparator) -> Self {
        Self {
            heap: Vec::new(),
            max_size: size,
            comparator,
        }
    }

    /// `lessThan(a, b)`: `comparator.compare(a, b) < 0`.
    fn less_than(&self, a: usize, b: usize) -> bool {
        (self.comparator)(&self.heap[a - 1], &self.heap[b - 1]) == Ordering::Less
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.heap.len()
    }

    /// `size() == 0`.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// `top()`: the least word, without removing it.
    pub fn top(&self) -> Option<&SuggestWord> {
        self.heap.first()
    }

    /// `add(element)`.
    ///
    /// # Errors
    /// The heap array is full (Java: `ArrayIndexOutOfBoundsException`).
    /// Java sizes it `maxSize + 1`, but 2 for `maxSize` 0, so a queue of
    /// size 0 takes one `add`.
    pub fn add(&mut self, element: SuggestWord) -> Result<()> {
        let heap_len = if self.max_size == 0 {
            2
        } else {
            self.max_size + 1
        };
        if self.heap.len() + 1 >= heap_len {
            return Err(Error::IllegalArgument(format!(
                "Index {} out of bounds for length {heap_len}",
                self.heap.len() + 1
            )));
        }
        self.heap.push(element);
        self.up_heap(self.heap.len());
        Ok(())
    }

    /// `insertWithOverflow(element)`: adds while there is room, else replaces
    /// the top when the newcomer is greater. Returns what did not fit: the
    /// newcomer, the evicted top, or `None`.
    pub fn insert_with_overflow(&mut self, element: SuggestWord) -> Option<SuggestWord> {
        if self.heap.len() < self.max_size {
            self.heap.push(element);
            self.up_heap(self.heap.len());
            return None;
        }
        if self.heap.is_empty() || (self.comparator)(&self.heap[0], &element) != Ordering::Less {
            return Some(element);
        }
        let evicted = std::mem::replace(&mut self.heap[0], element);
        self.down_heap(1);
        Some(evicted)
    }

    /// `pop()`: removes and answers the least word.
    pub fn pop(&mut self) -> Option<SuggestWord> {
        if self.heap.is_empty() {
            return None;
        }
        let last = self.heap.len() - 1;
        self.heap.swap(0, last);
        let result = self.heap.pop();
        if !self.heap.is_empty() {
            self.down_heap(1);
        }
        result
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// `upHeap(i)`: Java moves the parents down a hole and drops the node
    /// in; swapping along the same path leaves the same heap.
    fn up_heap(&mut self, mut i: usize) {
        let mut j = i >> 1;
        while j > 0 && self.less_than(i, j) {
            self.heap.swap(i - 1, j - 1);
            i = j;
            j >>= 1;
        }
    }

    /// `downHeap(i)`, by swaps as [`Self::up_heap`].
    fn down_heap(&mut self, mut i: usize) {
        let size = self.heap.len();
        loop {
            let mut j = i << 1;
            let k = j + 1;
            if k <= size && self.less_than(k, j) {
                j = k;
            }
            if j > size || !self.less_than(j, i) {
                break;
            }
            self.heap.swap(i - 1, j - 1);
            i = j;
        }
    }
}

/// `SuggestMode`: when a spell checker offers suggestions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuggestMode {
    /// `SUGGEST_WHEN_NOT_IN_INDEX`: only for a term the index lacks.
    SuggestWhenNotInIndex,
    /// `SUGGEST_MORE_POPULAR`: only terms more frequent than the query term.
    SuggestMorePopular,
    /// `SUGGEST_ALWAYS`: whether or not the term is in the index.
    SuggestAlways,
}

/// `CombineSuggestion`: a suggestion made by joining or breaking query
/// terms, with the indexes of the original terms it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct CombineSuggestion {
    /// `suggestion()`.
    pub suggestion: SuggestWord,
    /// `originalTermIndexes()`.
    pub original_term_indexes: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_queue_refuses_add_and_an_empty_one_pops_nothing() {
        let mut q = SuggestWordQueue::new(1);
        assert!(q.is_empty());
        assert!(q.top().is_none());
        assert!(q.pop().is_none());
        q.add(SuggestWord::new("a", 1.0, 1)).unwrap();
        let err = q.add(SuggestWord::new("b", 1.0, 1)).unwrap_err();
        assert!(err
            .to_string()
            .contains("Index 2 out of bounds for length 2"));
        assert_eq!(q.top().unwrap().string, "a");
        q.clear();
        assert_eq!(q.size(), 0);
        assert!(format!("{q:?}").starts_with("SuggestWordQueue"));
    }

    #[test]
    fn a_zero_sized_queue_hands_every_word_back() {
        let mut q = SuggestWordQueue::with_comparator(0, frequency_comparator());
        let w = SuggestWord::new("a", 1.0, 1);
        assert_eq!(q.insert_with_overflow(w.clone()), Some(w.clone()));
        assert!(q.is_empty());
        // Java's heap array has room for one add
        q.add(w.clone()).unwrap();
        assert!(q.add(w).is_err());
    }

    #[test]
    fn a_combine_suggestion_keeps_its_term_indexes() {
        let c = CombineSuggestion {
            suggestion: SuggestWord::new("ab", 1.0, 2),
            original_term_indexes: vec![0, 1],
        };
        assert_eq!(c.clone(), c);
        assert_ne!(SuggestMode::SuggestAlways, SuggestMode::SuggestMorePopular);
    }
}
