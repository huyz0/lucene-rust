//! `org.apache.lucene.util.PriorityQueue`, shared by the classifiers' top-k
//! queues and `misc`'s `HighFreqTerms`: a binary heap whose pop order on equal
//! elements is the heap's own, as Java's.

/// `org.apache.lucene.util.PriorityQueue` with its `lessThan`: a binary heap
/// whose `pop` order on equal elements is the heap's, not insertion order.
pub(crate) struct JavaPriorityQueue<T, F: Fn(&T, &T) -> bool> {
    /// 1-based, as Java's.
    heap: Vec<Option<T>>,
    size: usize,
    max_size: usize,
    less_than: F,
}

impl<T, F: Fn(&T, &T) -> bool> JavaPriorityQueue<T, F> {
    /// `new PriorityQueue(maxSize)`. The heap grows as elements arrive
    /// rather than being allocated at `max_size` up front: a size read off
    /// a request (`FuzzyLikeThisQuery`'s `maxNumTerms`) must not allocate
    /// what it never fills.
    pub(crate) fn new(max_size: usize, less_than: F) -> Self {
        let heap = vec![None];
        Self {
            heap,
            size: 0,
            max_size,
            less_than,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    fn less(&self, a: usize, b: usize) -> bool {
        match (&self.heap[a], &self.heap[b]) {
            (Some(x), Some(y)) => (self.less_than)(x, y),
            _ => false,
        }
    }

    /// `add(element)`; the queue must have room.
    fn add(&mut self, element: T) {
        self.size += 1;
        if self.size == self.heap.len() {
            self.heap.push(Some(element));
        } else {
            self.heap[self.size] = Some(element);
        }
        self.up_heap(self.size);
    }

    /// `insertWithOverflow(element)`: added while there is room, else it
    /// replaces the least element unless it is less than it.
    pub(crate) fn insert_with_overflow(&mut self, element: T) {
        if self.size < self.max_size {
            self.add(element);
        } else if self.size > 0 {
            let replaces = match &self.heap[1] {
                Some(top) => !(self.less_than)(&element, top),
                None => false,
            };
            if replaces {
                self.heap[1] = Some(element);
                self.down_heap(1);
            }
        }
    }

    /// `top()`.
    pub(crate) fn top(&self) -> Option<&T> {
        self.heap.get(1).and_then(Option::as_ref)
    }

    /// `pop()`.
    pub(crate) fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        let result = self.heap[1].take();
        self.heap.swap(1, self.size);
        self.size -= 1;
        self.down_heap(1);
        result
    }

    fn up_heap(&mut self, orig: usize) {
        let mut i = orig;
        let mut j = i >> 1;
        while j > 0 && {
            let node = self.heap[i].as_ref();
            let parent = self.heap[j].as_ref();
            matches!((node, parent), (Some(n), Some(p)) if (self.less_than)(n, p))
        } {
            self.heap.swap(i, j);
            i = j;
            j >>= 1;
        }
    }

    fn down_heap(&mut self, start: usize) {
        let mut i = start;
        let mut j = i << 1;
        let k = j + 1;
        if k <= self.size && self.less(k, j) {
            j = k;
        }
        while j <= self.size && self.less(j, i) {
            self.heap.swap(i, j);
            i = j;
            j = i << 1;
            let k = j + 1;
            if k <= self.size && self.less(k, j) {
                j = k;
            }
        }
    }
}
