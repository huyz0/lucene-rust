//! Faiss's binary heaps over parallel value/id arrays (`faiss/utils/Heap.h`),
//! reproduced operation for operation so that equal distances order the same
//! way Faiss orders them, and `HNSW::MinimaxHeap`.

/// Which end of the order a heap keeps at its top.
pub(crate) trait Order {
    /// `a` sorts above `b` in the heap (`cmp2`).
    fn above(a: f32, b: f32, ia: i64, ib: i64) -> bool;
    /// `cmp(a, b)`: `a` is strictly worse than `b` (a result handler's
    /// threshold against a candidate).
    fn worse(a: f32, b: f32) -> bool;
    /// The value an empty slot holds.
    const NEUTRAL: f32;
}

/// Largest on top: the heap of the `k` smallest distances (`CMax`).
pub(crate) struct Max;
/// Smallest on top: the heap of the `k` largest similarities (`CMin`).
pub(crate) struct Min;

impl Order for Max {
    fn above(a: f32, b: f32, ia: i64, ib: i64) -> bool {
        a > b || (a == b && ia > ib)
    }
    fn worse(a: f32, b: f32) -> bool {
        a > b
    }
    const NEUTRAL: f32 = f32::INFINITY;
}

impl Order for Min {
    fn above(a: f32, b: f32, ia: i64, ib: i64) -> bool {
        a < b || (a == b && ia < ib)
    }
    fn worse(a: f32, b: f32) -> bool {
        a < b
    }
    const NEUTRAL: f32 = f32::NEG_INFINITY;
}

/// `heap_pop`: removes the top of the `k` elements of `val`/`ids`, leaving
/// the last slot free.
pub(crate) fn heap_pop<C: Order>(k: usize, val: &mut [f32], ids: &mut [i64]) {
    let (v, id) = (val[k - 1], ids[k - 1]);
    // One-based indexing, as in Faiss.
    let mut i = 1usize;
    loop {
        let i1 = i << 1;
        let i2 = i1 + 1;
        if i1 > k {
            break;
        }
        let take_left = i2 == k + 1 || C::above(val[i1 - 1], val[i2 - 1], ids[i1 - 1], ids[i2 - 1]);
        let c = if take_left { i1 } else { i2 };
        if C::above(v, val[c - 1], id, ids[c - 1]) {
            break;
        }
        val[i - 1] = val[c - 1];
        ids[i - 1] = ids[c - 1];
        i = c;
    }
    val[i - 1] = v;
    ids[i - 1] = id;
}

/// `heap_push`: adds `(v, id)` as the `k`-th element of a heap of `k - 1`.
pub(crate) fn heap_push<C: Order>(k: usize, val: &mut [f32], ids: &mut [i64], v: f32, id: i64) {
    let mut i = k;
    while i > 1 {
        let father = i >> 1;
        if !C::above(v, val[father - 1], id, ids[father - 1]) {
            break;
        }
        val[i - 1] = val[father - 1];
        ids[i - 1] = ids[father - 1];
        i = father;
    }
    val[i - 1] = v;
    ids[i - 1] = id;
}

/// `heap_replace_top`: replaces the top of the `k`-element heap with `(v, id)`.
pub(crate) fn heap_replace_top<C: Order>(
    k: usize,
    val: &mut [f32],
    ids: &mut [i64],
    v: f32,
    id: i64,
) {
    let mut i = 1usize;
    loop {
        let i1 = i << 1;
        let i2 = i1 + 1;
        if i1 > k {
            break;
        }
        let take_left = i2 == k + 1 || C::above(val[i1 - 1], val[i2 - 1], ids[i1 - 1], ids[i2 - 1]);
        let c = if take_left { i1 } else { i2 };
        if C::above(v, val[c - 1], id, ids[c - 1]) {
            break;
        }
        val[i - 1] = val[c - 1];
        ids[i - 1] = ids[c - 1];
        i = c;
    }
    val[i - 1] = v;
    ids[i - 1] = id;
}

/// A top-`k` collector: the result handler of every search here.
pub(crate) struct TopK<C: Order> {
    k: usize,
    val: Vec<f32>,
    ids: Vec<i64>,
    _order: std::marker::PhantomData<C>,
}

impl<C: Order> TopK<C> {
    pub(crate) fn new(k: usize) -> Self {
        TopK {
            k,
            val: vec![C::NEUTRAL; k],
            ids: vec![-1; k],
            _order: std::marker::PhantomData,
        }
    }

    /// The value a candidate must beat (`threshold`).
    pub(crate) fn threshold(&self) -> f32 {
        if self.k == 0 {
            C::NEUTRAL
        } else {
            self.val[0]
        }
    }

    /// Offers `(v, id)`; whether it was kept.
    pub(crate) fn add(&mut self, v: f32, id: i64) -> bool {
        if self.k > 0 && C::worse(self.val[0], v) {
            heap_replace_top::<C>(self.k, &mut self.val, &mut self.ids, v, id);
            true
        } else {
            false
        }
    }

    /// `heap_reorder`: the kept results best first.
    pub(crate) fn finish(mut self) -> Vec<(i64, f32)> {
        let k = self.k;
        let mut written = 0usize;
        for i in 0..k {
            let (v, id) = (self.val[0], self.ids[0]);
            if id < 0 {
                break;
            }
            heap_pop::<C>(k - i, &mut self.val, &mut self.ids);
            self.val[k - written - 1] = v;
            self.ids[k - written - 1] = id;
            written += 1;
        }
        (k - written..k)
            .map(|i| (self.ids[i], self.val[i]))
            .collect()
    }
}

/// `HNSW::MinimaxHeap`: a bounded pool of candidates that yields its minimum
/// and drops its maximum. As in Faiss, a popped minimum leaves a hole (id
/// `-1`) that keeps its distance until the slot is evicted, and
/// [`Self::count_below`] counts holes too.
pub(crate) struct Minimax {
    n: usize,
    k: usize,
    nvalid: usize,
    ids: Vec<i64>,
    dis: Vec<f32>,
}

impl Minimax {
    pub(crate) fn new(n: usize) -> Self {
        Minimax {
            n,
            k: 0,
            nvalid: 0,
            ids: vec![0; n],
            dis: vec![0.0; n],
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.nvalid
    }

    pub(crate) fn push(&mut self, id: i64, v: f32) {
        if self.k == self.n {
            if v >= self.dis[0] {
                return;
            }
            if self.ids[0] != -1 {
                self.nvalid -= 1;
            }
            heap_pop::<Max>(self.k, &mut self.dis, &mut self.ids);
            self.k -= 1;
        }
        self.k += 1;
        heap_push::<Max>(self.k, &mut self.dis, &mut self.ids, v, id);
        self.nvalid += 1;
    }

    /// Removes and returns the nearest candidate, an O(k) scan.
    pub(crate) fn pop_min(&mut self) -> Option<(i64, f32)> {
        let mut imin = None;
        for i in (0..self.k).rev() {
            if self.ids[i] == -1 {
                continue;
            }
            match imin {
                None => imin = Some(i),
                Some(m) if self.dis[i] < self.dis[m] => imin = Some(i),
                _ => {}
            }
        }
        let m = imin?;
        let out = (self.ids[m], self.dis[m]);
        self.ids[m] = -1;
        self.nvalid -= 1;
        Some(out)
    }

    /// Slots (holes included) with a distance strictly below `thresh`.
    pub(crate) fn count_below(&self, thresh: f32) -> usize {
        self.dis[..self.k].iter().filter(|&&d| d < thresh).count()
    }
}
