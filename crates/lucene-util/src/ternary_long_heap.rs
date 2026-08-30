//! Port of `org.apache.lucene.util.TernaryLongHeap`.
//!
//! An `org.apache.lucene.util` class, used by `util/hnsw`'s `NeighborQueue`
//! and by `UpdateGraphsUtils.computeJoinSet`. Neither is a codec concern, so
//! neither is this.

/// Port of `org.apache.lucene.util.TernaryLongHeap`: a 1-based, arity-3 min
/// heap of `i64`.
///
/// Ported rather than replaced with `BinaryHeap` for one reason:
/// `NeighborQueue.nodes()` hands the raw heap array to
/// `HnswGraphBuilder`, which uses it as the entry-point set for the next
/// level down. The *contents* would be the same under any heap, but the
/// *order* would not, and that order feeds `scoreEntryPoints`'s collect
/// order. Matching Java here is what lets the differential test assert
/// "identical results", not merely "similar recall".
#[derive(Debug, Clone)]
pub struct TernaryLongHeap {
    initial_capacity: usize,
    /// 1-based: index 0 is unused, exactly as Java's.
    heap: Vec<i64>,
    size: usize,
}

const ARITY: usize = 3;

impl TernaryLongHeap {
    pub fn new(initial_capacity: usize) -> Self {
        assert!(initial_capacity >= 1, "initialCapacity must be > 0");
        TernaryLongHeap {
            initial_capacity,
            heap: vec![0; initial_capacity + 1],
            size: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn clear(&mut self) {
        self.size = 0;
    }

    pub fn top(&self) -> i64 {
        self.heap[1]
    }

    /// Java's `get(i)`, `i` in `1..=size`: heap-array order, not sorted order.
    pub fn get(&self, i: usize) -> i64 {
        self.heap[i]
    }

    pub fn push(&mut self, element: i64) {
        self.size += 1;
        if self.size == self.heap.len() {
            self.heap.resize((self.size * 3).div_ceil(2) + 1, 0);
        }
        self.heap[self.size] = element;
        Self::up_heap(&mut self.heap, self.size);
    }

    pub fn insert_with_overflow(&mut self, value: i64) -> bool {
        if self.size >= self.initial_capacity {
            if value < self.heap[1] {
                return false;
            }
            self.heap[1] = value;
            Self::down_heap(&mut self.heap, 1, self.size);
            return true;
        }
        self.push(value);
        true
    }

    pub fn pop(&mut self) -> i64 {
        assert!(self.size > 0, "The heap is empty");
        let result = self.heap[1];
        self.heap[1] = self.heap[self.size];
        self.size -= 1;
        Self::down_heap(&mut self.heap, 1, self.size);
        result
    }

    fn up_heap(heap: &mut [i64], mut i: usize) {
        let value = heap[i];
        while i > 1 {
            let parent = ((i - 2) / ARITY) + 1;
            let parent_val = heap[parent];
            if value >= parent_val {
                break;
            }
            heap[i] = parent_val;
            i = parent;
        }
        heap[i] = value;
    }

    fn down_heap(heap: &mut [i64], mut i: usize, size: usize) {
        if size == 0 {
            return;
        }
        let value = heap[i];
        loop {
            let first_child = ARITY * (i - 1) + 2;
            if first_child > size {
                break;
            }
            let last_child = (first_child + ARITY - 1).min(size);
            let mut best = first_child;
            let mut best_val = heap[first_child];
            for (offset, &v) in heap[first_child + 1..=last_child].iter().enumerate() {
                if v < best_val {
                    best_val = v;
                    best = first_child + 1 + offset;
                }
            }
            if best_val >= value {
                break;
            }
            heap[i] = best_val;
            i = best;
        }
        heap[i] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pops_in_ascending_order() {
        let mut heap = TernaryLongHeap::new(4);
        let input = [5i64, -3, 9, 0, 7, -8, 2, 2];
        for v in input {
            heap.push(v);
        }
        assert_eq!(heap.size(), input.len());
        assert_eq!(heap.top(), -8);
        let mut out = Vec::new();
        while heap.size() > 0 {
            out.push(heap.pop());
        }
        let mut sorted = input.to_vec();
        sorted.sort_unstable();
        assert_eq!(out, sorted);
    }

    #[test]
    fn insert_with_overflow_keeps_the_largest() {
        let mut heap = TernaryLongHeap::new(3);
        for v in [1i64, 2, 3] {
            assert!(heap.insert_with_overflow(v));
        }
        assert!(!heap.insert_with_overflow(0));
        assert!(heap.insert_with_overflow(9));
        let mut out = Vec::new();
        while heap.size() > 0 {
            out.push(heap.pop());
        }
        assert_eq!(out, vec![2, 3, 9]);
        heap.clear();
        assert_eq!(heap.size(), 0);
    }

    /// `get(i)` is heap-array order, not sorted order -- `NeighborQueue.nodes()`
    /// hands that raw array out and the HNSW builder uses it as an entry-point
    /// set, so the arity-3 layout is observable and must not be "improved".
    #[test]
    fn get_exposes_the_raw_arity_three_layout() {
        let mut heap = TernaryLongHeap::new(8);
        for v in [9i64, 8, 7, 6, 5, 4, 3] {
            heap.push(v);
        }
        assert_eq!(heap.get(1), heap.top());
        let raw: Vec<i64> = (1..=heap.size()).map(|i| heap.get(i)).collect();
        let mut sorted = raw.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![3, 4, 5, 6, 7, 8, 9]);
        assert_ne!(raw, sorted, "a heap array is not a sorted array");
    }

    #[test]
    #[should_panic(expected = "initialCapacity must be > 0")]
    fn a_zero_capacity_heap_is_refused() {
        let _ = TernaryLongHeap::new(0);
    }

    #[test]
    #[should_panic(expected = "The heap is empty")]
    fn popping_an_empty_heap_panics() {
        let mut heap = TernaryLongHeap::new(2);
        heap.pop();
    }
}
