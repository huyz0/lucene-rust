//! `TopOrdAndNumberQueue`, `TopOrdAndIntQueue` and `TopOrdAndFloatQueue`:
//! the bounded queues that keep a path's best children.
//!
//! The heap is Lucene's `PriorityQueue`, ported step for step
//! ([`PriorityQueue`]): the facet orderings are not total orders (a `NaN`
//! value is neither less nor greater than anything, so only the ordinal
//! decides), and only the same sift steps keep the same elements in the
//! same order.

use super::facet_result::FacetValue;

/// Lucene's `PriorityQueue`: a binary min-heap of at most `max_size`
/// elements under `less_than`, whose top is the least element.
pub(crate) struct PriorityQueue<T, F: Fn(&T, &T) -> bool> {
    /// `heap[1..=size]`, stored from index 0.
    heap: Vec<T>,
    max_size: usize,
    less_than: F,
}

impl<T, F: Fn(&T, &T) -> bool> PriorityQueue<T, F> {
    /// `new PriorityQueue(maxSize)`.
    pub(crate) fn new(max_size: usize, less_than: F) -> Self {
        PriorityQueue {
            heap: Vec::with_capacity(max_size.min(1 << 16)),
            max_size,
            less_than,
        }
    }

    /// `size()`.
    pub(crate) fn size(&self) -> usize {
        self.heap.len()
    }

    /// `top()`: the least element.
    pub(crate) fn top(&self) -> Option<&T> {
        self.heap.first()
    }

    /// `top()` to modify in place, then [`Self::update_top`].
    pub(crate) fn top_mut(&mut self) -> Option<&mut T> {
        self.heap.first_mut()
    }

    /// `add(element)`.
    pub(crate) fn add(&mut self, element: T) {
        self.heap.push(element);
        let last = self.heap.len();
        self.up_heap(last);
    }

    /// `insertWithOverflow(element)`: adds `element` while there is room;
    /// otherwise replaces the top when the top is less than it. Returns what
    /// fell out (`element` itself when it did not get in).
    pub(crate) fn insert_with_overflow(&mut self, element: T) -> Option<T> {
        if self.heap.len() < self.max_size {
            self.add(element);
            None
        } else if !self.heap.is_empty() && (self.less_than)(&self.heap[0], &element) {
            let old = std::mem::replace(&mut self.heap[0], element);
            self.down_heap(1);
            Some(old)
        } else {
            Some(element)
        }
    }

    /// `pop()`: removes and returns the least element.
    pub(crate) fn pop(&mut self) -> Option<T> {
        if self.heap.is_empty() {
            return None;
        }
        let result = self.heap.swap_remove(0);
        if !self.heap.is_empty() {
            self.down_heap(1);
        }
        Some(result)
    }

    /// `updateTop()`: restores the heap after the top changed.
    pub(crate) fn update_top(&mut self) {
        if !self.heap.is_empty() {
            self.down_heap(1);
        }
    }

    /// `lessThan(heap[a], heap[b])`, positions 1-based.
    fn less(&self, a: usize, b: usize) -> bool {
        (self.less_than)(
            &self.heap[a.saturating_sub(1)],
            &self.heap[b.saturating_sub(1)],
        )
    }

    /// Swaps 1-based positions `a` and `b`.
    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a.saturating_sub(1), b.saturating_sub(1));
    }

    /// `upHeap(origPos)`, positions 1-based: the hole Java moves up is a
    /// chain of swaps here, which ends with every element where Java's does.
    fn up_heap(&mut self, orig_pos: usize) {
        let mut i = orig_pos;
        let mut j = i >> 1;
        while j > 0 && self.less(i, j) {
            self.swap(i, j);
            i = j;
            j >>= 1;
        }
    }

    /// `downHeap(i)`, positions 1-based.
    fn down_heap(&mut self, mut i: usize) {
        let size = self.heap.len();
        let mut j = i.saturating_mul(2);
        let mut k = j.saturating_add(1);
        if k <= size && self.less(k, j) {
            j = k;
        }
        while j <= size && self.less(j, i) {
            self.swap(i, j);
            i = j;
            j = i.saturating_mul(2);
            k = j.saturating_add(1);
            if k <= size && self.less(k, j) {
                j = k;
            }
        }
    }
}

/// `TopOrdAndNumberQueue.OrdAndValue`: an ordinal and its value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrdAndValue {
    pub ord: i32,
    pub value: FacetValue,
}

/// `OrdAndInt.lessThan` / `OrdAndFloat.lessThan`: the lesser value, then the
/// greater ordinal.
fn ord_and_value_less_than(a: &OrdAndValue, b: &OrdAndValue) -> bool {
    let (lt, gt) = match (a.value, b.value) {
        (FacetValue::Int(x), FacetValue::Int(y)) => (x < y, x > y),
        (x, y) => {
            let (x, y) = (x.float_value(), y.float_value());
            (x < y, x > y)
        }
    };
    if lt {
        return true;
    }
    if gt {
        return false;
    }
    a.ord > b.ord
}

type LessThan = fn(&OrdAndValue, &OrdAndValue) -> bool;

/// `TopOrdAndNumberQueue`: the `top_n` best [`OrdAndValue`]s -- greatest
/// value first, the lesser ordinal winning a tie. [`Self::int`] is
/// `TopOrdAndIntQueue`, [`Self::float`] `TopOrdAndFloatQueue`; they differ
/// only in the values they hold.
pub struct TopOrdAndNumberQueue {
    queue: PriorityQueue<OrdAndValue, LessThan>,
}

impl TopOrdAndNumberQueue {
    /// `new TopOrdAndIntQueue(topN)`.
    pub fn int(top_n: usize) -> Self {
        Self::new(top_n)
    }

    /// `new TopOrdAndFloatQueue(topN)`.
    pub fn float(top_n: usize) -> Self {
        Self::new(top_n)
    }

    fn new(top_n: usize) -> Self {
        TopOrdAndNumberQueue {
            queue: PriorityQueue::new(top_n, ord_and_value_less_than as LessThan),
        }
    }

    /// `insertWithOverflow(ordAndValue)`.
    pub fn insert_with_overflow(&mut self, element: OrdAndValue) -> Option<OrdAndValue> {
        self.queue.insert_with_overflow(element)
    }

    /// `pop()`: the worst element kept.
    pub fn pop(&mut self) -> Option<OrdAndValue> {
        self.queue.pop()
    }

    /// `top()`: the worst element kept.
    pub fn top(&self) -> Option<&OrdAndValue> {
        self.queue.top()
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.queue.size()
    }

    /// The least value kept, once the queue is full and an int queue: a
    /// smaller int can never get in.
    #[inline]
    pub(crate) fn full_int_bottom(&self) -> Option<i32> {
        if self.queue.size() < self.queue.max_size {
            return None;
        }
        match self.queue.top()?.value {
            FacetValue::Int(v) => Some(v),
            FacetValue::Float(_) => None,
        }
    }
}

#[cfg(test)]
mod tests;
