//! `org.apache.lucene.search.suggest.InputIterator` and its wrappers: the
//! `(term, weight, payload, contexts)` stream a suggester is built from, the
//! buffering and sorting adapters, and the in-memory form of
//! `SortedInputIterator`.
//!
//! Java sorts on disk with `OfflineSorter` once the input outgrows memory. The
//! ordering is the same here, `(term, weight)` by unsigned bytes then weight
//! ascending, but the sort runs in memory: a suggester's input is a
//! `Vec` in this port, not a temp file. `Set<BytesRef> contexts` is a
//! `BTreeSet` so that iteration order is deterministic.

use std::collections::BTreeSet;

/// `InputIterator`: a stream of suggestion entries. Each call to
/// [`next`](Self::next) advances to the next entry; the accessors describe the
/// entry it last returned.
pub trait InputIterator {
    /// `next()`: the next term, or `None` at the end.
    fn next(&mut self) -> Option<Vec<u8>>;

    /// `weight()`: the entry's weight (its popularity).
    fn weight(&self) -> i64;

    /// `payload()`: the entry's payload, when [`has_payloads`](Self::has_payloads).
    fn payload(&self) -> Option<&[u8]>;

    /// `hasPayloads()`.
    fn has_payloads(&self) -> bool;

    /// `contexts()`: the entry's context set, when
    /// [`has_contexts`](Self::has_contexts).
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>>;

    /// `hasContexts()`.
    fn has_contexts(&self) -> bool;
}

/// `InputIterator.InputIteratorWrapper`: a plain term stream, weight 1, no
/// payloads, no contexts. `InputIterator.EMPTY` is `input_empty()`.
#[derive(Debug, Clone)]
pub struct InputIteratorWrapper<I> {
    wrapped: I,
}

impl<I> InputIteratorWrapper<I> {
    /// `new InputIteratorWrapper(BytesRefIterator)`.
    pub fn new(wrapped: I) -> Self {
        Self { wrapped }
    }
}

impl<I: Iterator<Item = Vec<u8>>> InputIterator for InputIteratorWrapper<I> {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.wrapped.next()
    }

    fn weight(&self) -> i64 {
        1
    }

    fn payload(&self) -> Option<&[u8]> {
        None
    }

    fn has_payloads(&self) -> bool {
        false
    }

    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        None
    }

    fn has_contexts(&self) -> bool {
        false
    }
}

/// `InputIterator.EMPTY`: a stream with no entries.
pub fn input_empty() -> InputIteratorWrapper<std::iter::Empty<Vec<u8>>> {
    InputIteratorWrapper::new(std::iter::empty())
}

/// `BufferedInputIterator`: drains its source at construction, then replays the
/// entries in order, so a suggester that needs two passes can read them twice.
#[derive(Debug, Clone)]
pub struct BufferedInputIterator {
    entries: Vec<Vec<u8>>,
    payloads: Vec<Vec<u8>>,
    context_sets: Vec<Option<BTreeSet<Vec<u8>>>>,
    freqs: Vec<i64>,
    /// `curPos`: the index of the current entry, `-1` before the first `next`.
    cur: isize,
    has_payloads: bool,
    has_contexts: bool,
}

impl BufferedInputIterator {
    /// `new BufferedInputIterator(source)`: reads every entry of `source`.
    pub fn new(source: &mut dyn InputIterator) -> Self {
        let has_payloads = source.has_payloads();
        let has_contexts = source.has_contexts();
        let mut out = Self {
            entries: Vec::new(),
            payloads: Vec::new(),
            context_sets: Vec::new(),
            freqs: Vec::new(),
            cur: -1,
            has_payloads,
            has_contexts,
        };
        while let Some(term) = source.next() {
            out.entries.push(term);
            if has_payloads {
                // Java appends the payload unchecked; a null here fails later
                // in Java, so the empty payload stands in for it.
                out.payloads
                    .push(source.payload().map(<[u8]>::to_vec).unwrap_or_default());
            }
            if has_contexts {
                out.context_sets.push(source.contexts().cloned());
            }
            out.freqs.push(source.weight());
        }
        out
    }
}

impl InputIterator for BufferedInputIterator {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.cur += 1;
        usize::try_from(self.cur)
            .ok()
            .and_then(|i| self.entries.get(i))
            .cloned()
    }

    fn weight(&self) -> i64 {
        // Java reads freqs[curPos] with no bounds check; before the first
        // `next` or after the end that is an exception there, and 0 here.
        self.index()
            .and_then(|i| self.freqs.get(i))
            .copied()
            .unwrap_or(0)
    }

    fn payload(&self) -> Option<&[u8]> {
        if !self.has_payloads {
            return None;
        }
        self.index()
            .and_then(|i| self.payloads.get(i))
            .map(Vec::as_slice)
    }

    fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        if !self.has_contexts {
            return None;
        }
        self.index()
            .and_then(|i| self.context_sets.get(i))
            .and_then(Option::as_ref)
    }

    fn has_contexts(&self) -> bool {
        self.has_contexts
    }
}

impl BufferedInputIterator {
    fn index(&self) -> Option<usize> {
        usize::try_from(self.cur).ok()
    }
}

/// One buffered record of a [`SortedInputIterator`].
#[derive(Debug, Clone)]
struct Entry {
    term: Vec<u8>,
    weight: i64,
    payload: Vec<u8>,
    contexts: Option<BTreeSet<Vec<u8>>>,
}

/// `SortedInputIterator`: drains its source, sorts by term (then by weight,
/// the tie-break Java's `tieBreakByCostComparator` uses), and replays in that
/// order.
#[derive(Debug, Clone)]
pub struct SortedInputIterator {
    entries: std::vec::IntoIter<Entry>,
    current: Option<Entry>,
    has_payloads: bool,
    has_contexts: bool,
}

impl SortedInputIterator {
    /// `new SortedInputIterator(tempDir, prefix, source)` with Java's natural
    /// order on terms: unsigned bytes, a proper prefix first.
    pub fn new(source: &mut dyn InputIterator) -> Self {
        Self::with_comparator(source, <[u8]>::cmp)
    }

    /// The `Comparator<BytesRef>` overload.
    pub fn with_comparator(
        source: &mut dyn InputIterator,
        cmp: impl Fn(&[u8], &[u8]) -> std::cmp::Ordering,
    ) -> Self {
        let has_payloads = source.has_payloads();
        let has_contexts = source.has_contexts();
        let mut entries = Vec::new();
        while let Some(term) = source.next() {
            entries.push(Entry {
                term,
                weight: source.weight(),
                payload: if has_payloads {
                    source.payload().map(<[u8]>::to_vec).unwrap_or_default()
                } else {
                    Vec::new()
                },
                contexts: if has_contexts {
                    source.contexts().cloned()
                } else {
                    None
                },
            });
        }
        // Stable, like the sort it stands in for; equal (term, weight) pairs
        // keep their input order.
        entries.sort_by(|a, b| cmp(&a.term, &b.term).then(a.weight.cmp(&b.weight)));
        Self {
            entries: entries.into_iter(),
            current: None,
            has_payloads,
            has_contexts,
        }
    }
}

impl InputIterator for SortedInputIterator {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.current = self.entries.next();
        self.current.as_ref().map(|e| e.term.clone())
    }

    fn weight(&self) -> i64 {
        self.current.as_ref().map_or(0, |e| e.weight)
    }

    fn payload(&self) -> Option<&[u8]> {
        if !self.has_payloads {
            return None;
        }
        Some(
            self.current
                .as_ref()
                .map_or(&[][..], |e| e.payload.as_slice()),
        )
    }

    fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        self.current.as_ref().and_then(|e| e.contexts.as_ref())
    }

    fn has_contexts(&self) -> bool {
        self.has_contexts
    }
}

/// `java.util.Random`, as much of it as `UnsortedInputIterator` uses: the
/// 48-bit linear congruential generator and `nextInt(bound)`.
#[derive(Debug, Clone)]
pub struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    const MULTIPLIER: u64 = 0x5DEECE66D;
    const MASK: u64 = (1 << 48) - 1;

    /// `new Random(seed)`.
    pub fn new(seed: i64) -> Self {
        Self {
            seed: (seed as u64 ^ Self::MULTIPLIER) & Self::MASK,
        }
    }

    fn next_bits(&mut self, bits: u32) -> i32 {
        self.seed = self.seed.wrapping_mul(Self::MULTIPLIER).wrapping_add(0xB) & Self::MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// `nextInt(bound)`; a `bound` below 1 is refused, as Java's
    /// `IllegalArgumentException`.
    pub fn next_int(&mut self, bound: i32) -> lucene_store::Result<i32> {
        if bound <= 0 {
            return Err(lucene_store::Error::IllegalArgument(
                "bound must be positive".into(),
            ));
        }
        Ok(self.next_int_positive(bound))
    }

    fn next_int_positive(&mut self, bound: i32) -> i32 {
        let mut r = self.next_bits(31);
        let m = bound.wrapping_sub(1);
        if bound & m == 0 {
            return ((i64::from(bound) * i64::from(r)) >> 31) as i32;
        }
        let mut u = r;
        loop {
            r = u % bound;
            if u.wrapping_sub(r).wrapping_add(m) >= 0 {
                return r;
            }
            u = self.next_bits(31);
        }
    }
}

/// `UnsortedInputIterator`: a [`BufferedInputIterator`] replayed in a random
/// order. Java seeds `new Random()` from the clock; [`new`](Self::new) does the
/// same and [`with_seed`](Self::with_seed) fixes the shuffle.
#[derive(Debug, Clone)]
pub struct UnsortedInputIterator {
    inner: BufferedInputIterator,
    ords: Vec<usize>,
}

impl UnsortedInputIterator {
    /// `new UnsortedInputIterator(source)`.
    pub fn new(source: &mut dyn InputIterator) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as i64);
        Self::with_seed(source, seed)
    }

    /// `new UnsortedInputIterator(source)` with `new Random(seed)`: the same
    /// swap-with-a-random-position shuffle as Java.
    pub fn with_seed(source: &mut dyn InputIterator, seed: i64) -> Self {
        let inner = BufferedInputIterator::new(source);
        let n = inner.entries.len();
        let mut ords: Vec<usize> = (0..n).collect();
        let mut random = JavaRandom::new(seed);
        for i in 0..n {
            let pos = random.next_int_positive(n as i32) as usize;
            ords.swap(i, pos);
        }
        Self { inner, ords }
    }

    fn ord(&self) -> Option<usize> {
        usize::try_from(self.inner.cur)
            .ok()
            .and_then(|i| self.ords.get(i))
            .copied()
    }
}

impl InputIterator for UnsortedInputIterator {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.inner.cur += 1;
        self.ord().map(|o| self.inner.entries[o].clone())
    }

    fn weight(&self) -> i64 {
        self.ord().map_or(0, |o| self.inner.freqs[o])
    }

    fn payload(&self) -> Option<&[u8]> {
        if !self.inner.has_payloads {
            return None;
        }
        self.ord()
            .and_then(|o| self.inner.payloads.get(o))
            .map(Vec::as_slice)
    }

    fn has_payloads(&self) -> bool {
        self.inner.has_payloads
    }

    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        if !self.inner.has_contexts {
            return None;
        }
        self.ord()
            .and_then(|o| self.inner.context_sets.get(o))
            .and_then(Option::as_ref)
    }

    fn has_contexts(&self) -> bool {
        self.inner.has_contexts
    }
}

#[cfg(test)]
mod tests;
