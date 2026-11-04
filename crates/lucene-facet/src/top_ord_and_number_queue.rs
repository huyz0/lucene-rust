//! `TopOrdAndNumberQueue`, `TopOrdAndIntQueue` and `TopOrdAndFloatQueue`:
//! the bounded queues that keep a path's best children.
//!
//! The heap is Lucene's `PriorityQueue`, ported step for step
//! ([`PriorityQueue`]): the facet orderings are not total orders (a `NaN`
//! value is neither less nor greater than anything, so only the ordinal
//! decides), and only the same sift steps keep the same elements in the
//! same order.

use super::facet_result::FacetValue;

/// A queue's `lessThan`: a closure, or a zero-sized ordering such as
/// [`OrdLess`] that a queue type can name (so the comparison inlines rather
/// than going through a function pointer).
pub(crate) trait LessThan<T> {
    fn less_than(&self, a: &T, b: &T) -> bool;
}

impl<T, F: Fn(&T, &T) -> bool> LessThan<T> for F {
    #[inline(always)]
    fn less_than(&self, a: &T, b: &T) -> bool {
        self(a, b)
    }
}

/// Lucene's `PriorityQueue`: a binary min-heap of at most `max_size`
/// elements under `less_than`, whose top is the least element.
///
/// Java's 1-based position `p` is index `p - 1` here (children `2i + 1` and
/// `2i + 2`, parent `(i - 1) / 2`): the same tree, so the same comparisons in
/// the same order and every element ends where Java's does.
pub(crate) struct PriorityQueue<T, F: LessThan<T>> {
    heap: Vec<T>,
    max_size: usize,
    less_than: F,
}

impl<T, F: LessThan<T>> PriorityQueue<T, F> {
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
        self.up_heap(self.heap.len().wrapping_sub(1));
    }

    /// `insertWithOverflow(element)`: adds `element` while there is room;
    /// otherwise replaces the top when the top is less than it. Returns what
    /// fell out (`element` itself when it did not get in).
    #[inline]
    pub(crate) fn insert_with_overflow(&mut self, element: T) -> Option<T> {
        if self.heap.len() < self.max_size {
            self.add(element);
            return None;
        }
        match self.heap.first_mut() {
            Some(top) if self.less_than.less_than(top, &element) => {
                let old = std::mem::replace(top, element);
                self.down_heap();
                Some(old)
            }
            _ => Some(element),
        }
    }

    /// `pop()`: removes and returns the least element.
    pub(crate) fn pop(&mut self) -> Option<T> {
        if self.heap.is_empty() {
            return None;
        }
        let result = self.heap.swap_remove(0);
        self.down_heap();
        Some(result)
    }

    /// `updateTop()`: restores the heap after the top changed.
    pub(crate) fn update_top(&mut self) {
        self.down_heap();
    }

    /// `upHeap(origPos)`: the hole Java moves up is a chain of swaps here,
    /// which ends with every element where Java's does.
    fn up_heap(&mut self, mut i: usize) {
        let heap = self.heap.as_mut_slice();
        while i > 0 {
            let j = i.wrapping_sub(1) / 2;
            if !self.less_than.less_than(&heap[i], &heap[j]) {
                break;
            }
            heap.swap(i, j);
            i = j;
        }
    }

    /// `downHeap(1)`: the top sifted down, the lesser child (the right one
    /// only when strictly less, as Java's) taking its place while it is less.
    fn down_heap(&mut self) {
        let heap = self.heap.as_mut_slice();
        let size = heap.len();
        let mut i = 0usize;
        loop {
            // No overflow: `i < size`, and a `Vec` holds under `isize::MAX`.
            let mut j = i.wrapping_mul(2).wrapping_add(1);
            if j >= size {
                break;
            }
            let k = j.wrapping_add(1);
            if k < size && self.less_than.less_than(&heap[k], &heap[j]) {
                j = k;
            }
            if !self.less_than.less_than(&heap[j], &heap[i]) {
                break;
            }
            heap.swap(i, j);
            i = j;
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
#[inline(always)]
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

/// [`ord_and_value_less_than`] as a type a queue can name.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OrdLess;

impl LessThan<OrdAndValue> for OrdLess {
    #[inline(always)]
    fn less_than(&self, a: &OrdAndValue, b: &OrdAndValue) -> bool {
        ord_and_value_less_than(a, b)
    }
}

/// `TopOrdAndNumberQueue`: the `top_n` best [`OrdAndValue`]s -- greatest
/// value first, the lesser ordinal winning a tie. [`Self::int`] is
/// `TopOrdAndIntQueue`, [`Self::float`] `TopOrdAndFloatQueue`; they differ
/// only in the values they hold.
pub struct TopOrdAndNumberQueue {
    queue: PriorityQueue<OrdAndValue, OrdLess>,
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
            queue: PriorityQueue::new(top_n, OrdLess),
        }
    }

    /// `insertWithOverflow(ordAndValue)`.
    #[inline]
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
