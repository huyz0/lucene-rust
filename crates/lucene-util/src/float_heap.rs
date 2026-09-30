//! Port of `org.apache.lucene.util.hnsw.FloatHeap` and `BlockingFloatHeap`:
//! a bounded min-heap of floats that keeps the `max_size` largest values
//! offered -- the global "minimum competitive score" behind
//! `MultiLeafKnnCollector`, shared between the leaves of one search.
//!
//! The sift loops are Java's, slot for slot (1-based array, strict `<`), so
//! the heap's layout -- and [`FloatHeap::heap`]'s order -- matches Lucene's
//! after any sequence of operations. `BlockingFloatHeap` is the same heap
//! behind a lock; Java's `ReentrantLock` becomes a [`std::sync::Mutex`].

use std::sync::Mutex;

/// The heap proper, shared by both types.
#[derive(Debug, Clone)]
struct Heap {
    max_size: usize,
    /// 1-based: `heap[0]` is unused, as in Java.
    heap: Vec<f32>,
    size: usize,
}

impl Heap {
    fn new(max_size: usize) -> Self {
        Heap {
            max_size,
            heap: vec![0.0; max_size + 1],
            size: 0,
        }
    }

    fn push(&mut self, element: f32) {
        self.size += 1;
        self.heap[self.size] = element;
        self.up_heap(self.size);
    }

    fn update_top(&mut self, value: f32) -> f32 {
        self.heap[1] = value;
        self.down_heap(1);
        self.heap[1]
    }

    fn poll(&mut self) -> Option<f32> {
        if self.size == 0 {
            return None;
        }
        let result = self.heap[1];
        self.heap[1] = self.heap[self.size];
        self.size -= 1;
        self.down_heap(1);
        Some(result)
    }

    fn down_heap(&mut self, mut i: usize) {
        let value = self.heap[i];
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= self.size && self.heap[k] < self.heap[j] {
            j = k;
        }
        while j <= self.size && self.heap[j] < value {
            self.heap[i] = self.heap[j];
            i = j;
            j = i << 1;
            k = j + 1;
            if k <= self.size && self.heap[k] < self.heap[j] {
                j = k;
            }
        }
        self.heap[i] = value;
    }

    fn up_heap(&mut self, orig_pos: usize) {
        let mut i = orig_pos;
        let value = self.heap[i];
        let mut j = i >> 1;
        while j > 0 && value < self.heap[j] {
            self.heap[i] = self.heap[j];
            i = j;
            j >>= 1;
        }
        self.heap[i] = value;
    }
}

/// `FloatHeap`: single-threaded.
#[derive(Debug, Clone)]
pub struct FloatHeap {
    inner: Heap,
}

impl FloatHeap {
    /// `new FloatHeap(maxSize)`.
    pub fn new(max_size: usize) -> Self {
        FloatHeap {
            inner: Heap::new(max_size),
        }
    }

    /// `offer(value)`: add `value` if the heap is not full or it beats the
    /// current minimum; returns whether it was kept.
    pub fn offer(&mut self, value: f32) -> bool {
        let h = &mut self.inner;
        if h.size >= h.max_size {
            // A zero-capacity heap keeps nothing (Java reads heap[1] of a
            // 1-slot array, i.e. the unused 0.0, and would overwrite slot 1
            // past its end; here that case is simply "not kept").
            if h.max_size == 0 || value < h.heap[1] {
                return false;
            }
            h.update_top(value);
            return true;
        }
        h.push(value);
        true
    }

    /// `getHeap()`: the stored values in heap order.
    pub fn heap(&self) -> Vec<f32> {
        self.inner.heap[1..=self.inner.size].to_vec()
    }

    /// `poll()`: remove and return the minimum; `None` when empty (Java's
    /// `IllegalStateException`).
    pub fn poll(&mut self) -> Option<f32> {
        self.inner.poll()
    }

    /// `peek()`: the minimum (Java returns slot 1 even when empty, i.e. a
    /// stale value or 0; here `None`).
    pub fn peek(&self) -> Option<f32> {
        (self.inner.size > 0).then(|| self.inner.heap[1])
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.inner.size
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.inner.size = 0;
    }
}

/// `BlockingFloatHeap`: the same heap, safe to share between threads.
#[derive(Debug)]
pub struct BlockingFloatHeap {
    inner: Mutex<Heap>,
}

impl BlockingFloatHeap {
    /// `new BlockingFloatHeap(maxSize)`. `max_size` must be positive (Java
    /// reads `heap[1]` unconditionally).
    pub fn new(max_size: usize) -> Self {
        assert!(max_size > 0, "maxSize must be positive");
        BlockingFloatHeap {
            inner: Mutex::new(Heap::new(max_size)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Heap> {
        // A panic while holding the lock leaves a well-formed heap (every
        // mutation completes before it can unwind), so poisoning is ignored.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `offer(value)`: add or replace the minimum; returns the new minimum.
    pub fn offer(&self, value: f32) -> f32 {
        let mut h = self.lock();
        if h.size < h.max_size {
            h.push(value);
        } else if value >= h.heap[1] {
            h.update_top(value);
        }
        h.heap[1]
    }

    /// `offer(values, len)`: offer `values` (sorted ascending), largest
    /// first, stopping at the first one that no longer competes; returns
    /// the new minimum.
    pub fn offer_all(&self, values: &[f32]) -> f32 {
        let mut h = self.lock();
        for &v in values.iter().rev() {
            if h.size < h.max_size {
                h.push(v);
            } else if v >= h.heap[1] {
                h.update_top(v);
            } else {
                break;
            }
        }
        h.heap[1]
    }

    /// `poll()`: `None` when empty.
    pub fn poll(&self) -> Option<f32> {
        self.lock().poll()
    }

    /// `peek()`: slot 1, as Java (0 before anything was offered).
    pub fn peek(&self) -> f32 {
        self.lock().heap[1]
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.lock().size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_largest_and_polls_ascending() {
        let mut h = FloatHeap::new(3);
        for v in [5.0, 1.0, 3.0, 4.0, 0.5, 6.0] {
            h.offer(v);
        }
        assert_eq!(h.size(), 3);
        assert_eq!(h.peek(), Some(4.0));
        assert!(!h.offer(2.0));
        assert!(h.offer(4.5));
        let mut sorted = h.heap();
        sorted.sort_by(f32::total_cmp);
        assert_eq!(sorted, vec![4.5, 5.0, 6.0]);
        assert_eq!(h.poll(), Some(4.5));
        assert_eq!(h.poll(), Some(5.0));
        assert_eq!(h.poll(), Some(6.0));
        assert_eq!(h.poll(), None);
        assert_eq!(h.peek(), None);
        h.offer(1.0);
        h.clear();
        assert_eq!(h.size(), 0);
        let mut z = FloatHeap::new(0);
        assert!(!z.offer(1.0));
    }

    #[test]
    fn heap_layout_matches_java_sift_order() {
        // Java: offer 5,1,3 into a FloatHeap(4) -> heap [1,5,3]; offer 0 -> [0,1,3,5].
        let mut h = FloatHeap::new(4);
        for v in [5.0, 1.0, 3.0] {
            h.offer(v);
        }
        assert_eq!(h.heap(), vec![1.0, 5.0, 3.0]);
        h.offer(0.0);
        assert_eq!(h.heap(), vec![0.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn blocking_heap_offers_and_bulk_offers() {
        let h = BlockingFloatHeap::new(3);
        assert_eq!(h.peek(), 0.0);
        assert_eq!(h.offer(2.0), 2.0);
        assert_eq!(h.offer(1.0), 1.0);
        assert_eq!(h.offer(3.0), 1.0);
        assert_eq!(h.offer(0.5), 1.0); // does not compete
        assert_eq!(h.offer(1.5), 1.5);
        // Ascending batch, taken from the top until one fails to compete.
        assert_eq!(h.offer_all(&[0.1, 1.0, 2.5, 4.0]), 2.5);
        assert_eq!(h.size(), 3);
        assert_eq!(h.poll(), Some(2.5));
        assert_eq!(h.poll(), Some(3.0));
        assert_eq!(h.poll(), Some(4.0));
        assert_eq!(h.poll(), None);
        assert_eq!(h.offer_all(&[1.0, 2.0]), 1.0);
    }

    #[test]
    fn blocking_heap_is_shareable() {
        let h = std::sync::Arc::new(BlockingFloatHeap::new(10));
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let h = h.clone();
                std::thread::spawn(move || {
                    for i in 0..100 {
                        h.offer((t * 100 + i) as f32);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(h.size(), 10);
        assert_eq!(h.peek(), 390.0);
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn blocking_heap_needs_capacity() {
        BlockingFloatHeap::new(0);
    }
}
