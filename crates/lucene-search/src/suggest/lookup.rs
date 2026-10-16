//! `org.apache.lucene.search.suggest.Lookup` (M13 T13.3a): the suggester API
//! every concrete suggester implements, its `LookupResult`, the
//! `CHARSEQUENCE_COMPARATOR` that orders keys, and the `LookupPriorityQueue`
//! that keeps the top `n` results by weight.
//!
//! Keys are compared as UTF-16 code units, as Java's `CharSequence` is, so the
//! order of two keys that differ only in a supplementary character matches
//! Lucene's: `U+1F600` (surrogates `D83D DE00`) sorts before `U+FFFD`, which
//! code-point order would put the other way round.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use lucene_store::{DataInput, DataOutput};

use super::input::InputIterator;

/// `Lookup.CHARSEQUENCE_COMPARATOR`: the first differing UTF-16 code unit
/// decides; a proper prefix sorts first.
pub fn charsequence_compare(a: &str, b: &str) -> Ordering {
    let (mut x, mut y) = (a.encode_utf16(), b.encode_utf16());
    loop {
        match (x.next(), y.next()) {
            (Some(p), Some(q)) if p != q => return p.cmp(&q),
            (Some(_), Some(_)) => {}
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
        }
    }
}

/// `Lookup.LookupResult`: one suggestion. Ordered by `key` alone, as Java's
/// `compareTo` is.
#[derive(Debug, Clone, Default)]
pub struct LookupResult {
    pub key: String,
    /// Java's `Object highlightKey`: the markup-ready key, if the suggester
    /// produced one.
    pub highlight_key: Option<String>,
    pub value: i64,
    pub payload: Option<Vec<u8>>,
    /// Java's `Set<BytesRef> contexts`; ordered here for determinism.
    pub contexts: Option<BTreeSet<Vec<u8>>>,
}

impl LookupResult {
    /// `LookupResult(CharSequence key, long value)`.
    pub fn new(key: impl Into<String>, value: i64) -> Self {
        Self {
            key: key.into(),
            value,
            ..Self::default()
        }
    }

    /// `LookupResult(key, highlightKey, value, payload, contexts)`, the
    /// constructor every other Java overload delegates to.
    pub fn with_all(
        key: impl Into<String>,
        highlight_key: Option<String>,
        value: i64,
        payload: Option<Vec<u8>>,
        contexts: Option<BTreeSet<Vec<u8>>>,
    ) -> Self {
        Self {
            key: key.into(),
            highlight_key,
            value,
            payload,
            contexts,
        }
    }
}

impl PartialEq for LookupResult {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for LookupResult {}

impl PartialOrd for LookupResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LookupResult {
    fn cmp(&self, other: &Self) -> Ordering {
        charsequence_compare(&self.key, &other.key)
    }
}

/// `Lookup.LookupPriorityQueue`: a bounded min-heap on `value`, so the
/// smallest-weighted result is the one evicted when a bigger one arrives.
///
/// The heap is Java's `PriorityQueue` array-heap, operation for operation
/// (1-indexed, up-heap on insert, down-heap on pop and replace), so results
/// that tie on `value` come out in the same order Lucene's do.
#[derive(Debug)]
pub struct LookupPriorityQueue {
    /// Index 0 is unused, as in Java's 1-indexed heap.
    heap: Vec<LookupResult>,
    max_size: usize,
}

impl LookupPriorityQueue {
    /// `new LookupPriorityQueue(size)`.
    pub fn new(size: usize) -> Self {
        Self {
            heap: vec![LookupResult::default()],
            max_size: size,
        }
    }

    /// `size()`.
    pub fn len(&self) -> usize {
        self.heap.len() - 1
    }

    /// `size() == 0`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `insertWithOverflow`: keeps the `max_size` largest values. Returns the
    /// entry that did not fit (the rejected newcomer, or the evicted minimum),
    /// or `None` when nothing was dropped.
    pub fn insert_with_overflow(&mut self, element: LookupResult) -> Option<LookupResult> {
        if self.len() < self.max_size {
            self.add(element);
            return None;
        }
        if self.max_size == 0 {
            return Some(element);
        }
        if !less_than(&self.heap[1], &element) {
            return Some(element);
        }
        let evicted = std::mem::replace(&mut self.heap[1], element);
        self.down_heap(1);
        Some(evicted)
    }

    /// `pop()`: the smallest value, or `None` when empty.
    pub fn pop(&mut self) -> Option<LookupResult> {
        let size = self.len();
        if size == 0 {
            return None;
        }
        let last = self.heap.pop()?;
        if size == 1 {
            return Some(last);
        }
        let result = std::mem::replace(&mut self.heap[1], last);
        self.down_heap(1);
        Some(result)
    }

    /// `getResults()`: every entry, the largest value first.
    pub fn get_results(&mut self) -> Vec<LookupResult> {
        let size = self.len();
        let mut res = vec![LookupResult::default(); size];
        for i in (0..size).rev() {
            if let Some(r) = self.pop() {
                res[i] = r;
            }
        }
        res
    }

    fn add(&mut self, element: LookupResult) {
        self.heap.push(element);
        let last = self.len();
        self.up_heap(last);
    }

    fn up_heap(&mut self, mut i: usize) {
        let node = self.heap[i].clone();
        let mut j = i >> 1;
        while j > 0 && less_than(&node, &self.heap[j]) {
            self.heap[i] = self.heap[j].clone();
            i = j;
            j >>= 1;
        }
        self.heap[i] = node;
    }

    fn down_heap(&mut self, mut i: usize) {
        let size = self.len();
        let node = self.heap[i].clone();
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= size && less_than(&self.heap[k], &self.heap[j]) {
            j = k;
        }
        while j <= size && less_than(&self.heap[j], &node) {
            self.heap[i] = self.heap[j].clone();
            i = j;
            j <<= 1;
            k = j + 1;
            if k <= size && less_than(&self.heap[k], &self.heap[j]) {
                j = k;
            }
        }
        self.heap[i] = node;
    }
}

/// `LookupPriorityQueue.lessThan`: ordered by `value` alone.
fn less_than(a: &LookupResult, b: &LookupResult) -> bool {
    a.value < b.value
}

/// Java's `IllegalStateException(message)`: the store `Error` has no such
/// variant, so it travels as an `io::Error` of kind `Other` whose text names it.
pub fn illegal_state(message: impl Into<String>) -> lucene_store::Error {
    lucene_store::Error::Io(std::io::Error::other(format!(
        "IllegalStateException: {}",
        message.into()
    )))
}

/// Java's `UnsupportedOperationException(message)`, as an `io::Error` of kind
/// `Unsupported`.
pub fn unsupported_operation(message: impl Into<String>) -> lucene_store::Error {
    lucene_store::Error::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        message.into(),
    ))
}

/// `org.apache.lucene.search.suggest.Lookup`: a suggester built from an
/// [`InputIterator`] and asked for the top completions of a key.
pub trait Lookup {
    /// `getCount()`: the number of suggestions the suggester holds.
    fn get_count(&self) -> lucene_store::Result<u64>;

    /// `build(InputIterator)`: replaces whatever the suggester held.
    fn build(&mut self, input: &mut dyn InputIterator) -> lucene_store::Result<()>;

    /// `lookup(key, contexts, onlyMorePopular, num)`.
    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        only_more_popular: bool,
        num: usize,
    ) -> lucene_store::Result<Vec<LookupResult>>;

    /// `lookup(key, onlyMorePopular, num)`: the lookup with no contexts.
    fn lookup_no_contexts(
        &self,
        key: &str,
        only_more_popular: bool,
        num: usize,
    ) -> lucene_store::Result<Vec<LookupResult>> {
        self.lookup(key, None, only_more_popular, num)
    }

    /// `store(DataOutput)`: `false` when the suggester cannot be persisted.
    fn store(&self, output: &mut dyn DataOutput) -> lucene_store::Result<bool>;

    /// `load(DataInput)`: `false` when the bytes are not this suggester's.
    fn load(&mut self, input: &mut dyn DataInput) -> lucene_store::Result<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(key: &str, value: i64) -> LookupResult {
        LookupResult::new(key, value)
    }

    #[test]
    fn comparator_orders_by_utf16_units() {
        // U+1F600 is D83D DE00 in UTF-16, which is below U+FFFD; in code
        // point order it would be above.
        assert_eq!(
            charsequence_compare("\u{1F600}", "\u{FFFD}"),
            Ordering::Less
        );
        assert_eq!(charsequence_compare("ab", "abc"), Ordering::Less);
        assert_eq!(charsequence_compare("abc", "ab"), Ordering::Greater);
        assert_eq!(charsequence_compare("b", "a"), Ordering::Greater);
        assert_eq!(charsequence_compare("same", "same"), Ordering::Equal);
        assert_eq!(charsequence_compare("", ""), Ordering::Equal);
        assert_eq!(charsequence_compare("", "x"), Ordering::Less);
    }

    #[test]
    fn results_compare_by_key_only() {
        assert!(r("a", 9) < r("b", 1));
        assert_eq!(r("a", 1), r("a", 9));
        assert_eq!(r("a", 1).partial_cmp(&r("a", 2)), Some(Ordering::Equal));
        let full = LookupResult::with_all("k", Some("<k>".into()), 3, Some(vec![1]), None);
        assert_eq!(full.highlight_key.as_deref(), Some("<k>"));
        assert_eq!(full.payload.as_deref(), Some(&[1u8][..]));
        assert!(full.contexts.is_none());
    }

    #[test]
    fn queue_keeps_the_largest_values_in_descending_order() {
        let mut q = LookupPriorityQueue::new(3);
        assert!(q.is_empty());
        assert!(q.insert_with_overflow(r("a", 5)).is_none());
        assert!(q.insert_with_overflow(r("b", 1)).is_none());
        assert!(q.insert_with_overflow(r("c", 9)).is_none());
        // full: 1 is the smallest, so 1 is evicted for 7
        assert_eq!(q.insert_with_overflow(r("d", 7)).unwrap().key, "b");
        // 2 is smaller than the minimum (5), so it is the one rejected
        assert_eq!(q.insert_with_overflow(r("e", 2)).unwrap().key, "e");
        let got: Vec<(String, i64)> = q
            .get_results()
            .into_iter()
            .map(|x| (x.key, x.value))
            .collect();
        assert_eq!(got, vec![("c".into(), 9), ("d".into(), 7), ("a".into(), 5)]);
        assert!(q.is_empty());
        assert!(q.pop().is_none());
    }

    #[test]
    fn zero_capacity_rejects_everything() {
        let mut q = LookupPriorityQueue::new(0);
        assert_eq!(q.insert_with_overflow(r("x", 1)).unwrap().key, "x");
        assert!(q.get_results().is_empty());
    }

    #[test]
    fn pop_returns_the_smallest_first() {
        let mut q = LookupPriorityQueue::new(10);
        for (i, v) in [4i64, 8, 1, 9, 3, 7].iter().enumerate() {
            q.insert_with_overflow(r(&i.to_string(), *v));
        }
        assert_eq!(q.len(), 6);
        assert_eq!(q.pop().unwrap().value, 1);
        assert_eq!(q.pop().unwrap().value, 3);
        assert_eq!(q.len(), 4);
        let rest: Vec<i64> = q.get_results().iter().map(|x| x.value).collect();
        assert_eq!(rest, vec![9, 8, 7, 4]);
    }

    #[test]
    fn single_element_pop_empties_the_queue() {
        let mut q = LookupPriorityQueue::new(2);
        q.insert_with_overflow(r("only", 3));
        assert_eq!(q.pop().unwrap().key, "only");
        assert!(q.is_empty());
    }

    #[test]
    fn ties_leave_in_heap_order_as_java_does() {
        // Adds a,b,c,d (all value 1) leave the heap as [a,b,c,d]: up-heap
        // swaps only on a strict less-than, so ties do not move. get_results
        // then pops a, d, c, b and fills from the end, giving [b,c,d,a].
        let mut q = LookupPriorityQueue::new(4);
        for k in ["a", "b", "c", "d"] {
            q.insert_with_overflow(r(k, 1));
        }
        let keys: Vec<String> = q.get_results().into_iter().map(|x| x.key).collect();
        assert_eq!(keys, vec!["b", "c", "d", "a"]);
    }
}
