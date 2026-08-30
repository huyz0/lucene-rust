//! Per-segment SORTED_SET / SORTED ordinals mapped into one global ordinal
//! space — a port of `org.apache.lucene.index.OrdinalMap`.
//!
//! ## Why this is here and not in `lucene-index`
//!
//! Java's `OrdinalMap` lives in `lucene/core/src/java/org/apache/lucene/index/`
//! because doc-values *merging* uses it as well as faceting. In this port
//! nothing below `lucene-search` has a consumer for it yet (the merge path
//! rebuilds a segment's dictionary from scratch rather than mapping ordinals),
//! and faceting — [`crate::facets`] — is the one caller. Keeping it beside its
//! caller instead of speculatively parking it a crate down is a one-file move
//! if the merge path ever needs it.
//!
//! ## What it is for
//!
//! A SORTED_SET field's ordinals are **per segment**: each segment's terms
//! dictionary numbers its own terms 0..n, so ordinal 3 in segment 0 and
//! ordinal 3 in segment 1 are, in general, different terms. Summing raw
//! per-segment facet counts therefore conflates unrelated terms — this is the
//! defect [`crate::facets`]' module doc used to warn callers about and work
//! around by merging on the resolved label instead.
//!
//! `OrdinalMap` removes the workaround: it merge-sorts every segment's terms
//! into one global, still-sorted dictionary and records, for each segment,
//! `segmentOrd -> globalOrd`. Counting then becomes
//! `global[map.global_ord(seg, local)] += local_count`, which is exactly
//! `SortedSetDocValuesFacetCounts.countOneSegment`'s
//! `counts[(int) ordMap.get(ord)] += count`.
//!
//! ## Divergences from Java, deliberately
//!
//! - **Representation.** Java packs `segmentToGlobalOrds` as
//!   `PackedLongValues` deltas against `globalOrdDeltas`/`firstSegments`,
//!   tuned by an `acceptableOverheadRatio`, because it is holding this for a
//!   whole index in a JVM heap. This port stores a plain `Vec<i64>` per
//!   segment. The mapping is identical — pinned against real Lucene's own
//!   `OrdinalMap` output in `tests/facets_fixtures.rs` — and the space is a
//!   deliberate trade: a `Vec<i64>` is 8 bytes per *segment ordinal*, and the
//!   dictionaries this port faces are the ones a single `DirectoryReader`'s
//!   segments carry.
//! - **No segment reordering.** Java sorts segments by weight (descending
//!   unique-term count) so that the common terms' deltas land in the
//!   cheapest-to-pack segment. That ordering is invisible in
//!   `segmentToGlobalOrds` — global ordinals are assigned by *term* order, not
//!   segment order — so with a plain vector representation there is nothing
//!   for it to optimize, and it is not ported. [`OrdinalMap::first_segment`]
//!   consequently reports the lowest *input* segment index containing a term
//!   rather than the lowest weight-sorted one; that is documented on the
//!   method, and is the only place the difference is observable.
//! - **The input can be streamed, as Java's is.** `OrdinalMap.build` takes
//!   `TermsEnum[]` and never holds a dictionary, and so does
//!   [`OrdinalMap::build_streaming`], over any [`TermCursor`] --
//!   [`lucene_codecs::terms_dict::TermsCursor`] being the one that reads a
//!   real SORTED/SORTED_SET dictionary. [`OrdinalMap::build`] still takes
//!   materialized lists, for a caller that has them anyway, and now runs the
//!   same merge over a slice-backed cursor rather than a second algorithm.
//!
//!   **Measured** (`examples/ordinal_map_memory.rs`, 17-byte terms, Linux
//!   RSS with the allocation totals agreeing to within the allocator's own
//!   slack), before this change:
//!
//!   | shape | materialized input | the map itself | peak |
//!   |---|---|---|---|
//!   | 5 segments x 1 M terms, 1.2 M global | 267 MB | 51 MB | 319 MB |
//!   | 10 x 200 k, 380 k global | 107 MB | 20 MB | 127 MB |
//!   | 20 x 50 k, 161 k global | 53 MB | 10 MB | 63 MB |
//!
//!   So the input was **~5x the map** and ~84% of the peak. Streaming it
//!   removes that column outright: what `build_streaming` holds is one reused
//!   term buffer per segment plus the map. The `Vec<i64>`-rather-than-
//!   `PackedLongValues` divergence above is the 51 MB column, i.e. what is
//!   left once the larger half is gone. See
//!   `docs/sweep/m2/c38-allocation-shape.md`.
//! - **No `IndexReader.CacheKey` owner.** Java carries one so
//!   `DefaultSortedSetDocValuesReaderState` can cache the map against the
//!   reader it was built for. This port has no reader cache-key mechanism, so
//!   the caller owns the lifetime.

use std::cmp::Ordering;

/// One sub-enumerator of the k-way merge `OrdinalMap.build` runs — this
/// port's `TermsEnumIndex`.
///
/// Java's is a `TermsEnum`; this is the one method of it `OrdinalMap` uses
/// (`next()`, with `ord()` implied by the call count, because a doc-values
/// terms dictionary numbers its terms densely from zero in the order it
/// yields them). The returned slice only has to live until the next call, so
/// an implementation can hand back a reused buffer — which is what
/// [`lucene_codecs::terms_dict::TermsCursor`] does, and the reason a whole
/// dictionary never has to exist.
pub trait TermCursor {
    /// `TermsEnum.next()`: the next term in ascending byte order, or `None`
    /// at the end.
    fn next_term(&mut self) -> lucene_store::Result<Option<&[u8]>>;
}

impl TermCursor for lucene_codecs::terms_dict::TermsCursor<'_> {
    fn next_term(&mut self) -> lucene_store::Result<Option<&[u8]>> {
        lucene_codecs::terms_dict::TermsCursor::next_term(self)
    }
}

/// A [`TermCursor`] over an already-materialized term list — what
/// [`OrdinalMap::build`] runs the same merge over, so there is one algorithm
/// and not two.
struct SliceCursor<'a, T> {
    terms: &'a [T],
    next: usize,
}

impl<T: AsRef<[u8]>> TermCursor for SliceCursor<'_, T> {
    fn next_term(&mut self) -> lucene_store::Result<Option<&[u8]>> {
        let term = self.terms.get(self.next).map(AsRef::as_ref);
        if term.is_some() {
            self.next = self.next.saturating_add(1);
        }
        Ok(term)
    }
}

/// `OrdinalMap`'s `TermsEnumPriorityQueue`: a min-heap of *segment indices*,
/// ordered by each segment's current term and then by the index itself.
///
/// The keys live outside the heap, in the caller's `current` buffers, because
/// a streaming cursor's term is a borrow that only survives until its next
/// call — so the merge keeps one reused buffer per segment and the queue
/// orders indices into it. Ties breaking towards the lowest segment index is
/// what makes [`OrdinalMap::first_segment`] well-defined.
#[derive(Default)]
struct MergeQueue {
    heap: Vec<usize>,
}

impl MergeQueue {
    fn with_capacity(n: usize) -> Self {
        MergeQueue {
            heap: Vec::with_capacity(n),
        }
    }

    /// The segment whose current term is smallest.
    fn peek(&self) -> Option<usize> {
        self.heap.first().copied()
    }

    fn less(keys: &[Vec<u8>], a: usize, b: usize) -> bool {
        match keys[a].cmp(&keys[b]) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => a < b,
        }
    }

    fn push(&mut self, keys: &[Vec<u8>], segment: usize) {
        self.heap.push(segment);
        // ARITH: `child` starts at `heap.len() - 1` on a non-empty heap and
        // strictly decreases, so `(child - 1) / 2` is in range at every step.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let mut child = self.heap.len() - 1;
            while child > 0 {
                let parent = (child - 1) / 2;
                if Self::less(keys, self.heap[child], self.heap[parent]) {
                    self.heap.swap(child, parent);
                    child = parent;
                } else {
                    break;
                }
            }
        }
    }

    /// `PriorityQueue.updateTop()`: the top segment's key has changed in
    /// place (its cursor advanced), so restore the heap with **one** sift-down
    /// rather than the pop-then-push pair that would do the work twice. This
    /// is the merge's hot path -- it runs once per `(segment, distinct term)`.
    fn update_top(&mut self, keys: &[Vec<u8>]) {
        self.sift_down(keys, 0);
    }

    fn pop(&mut self, keys: &[Vec<u8>]) -> Option<usize> {
        let top = *self.heap.first()?;
        let last = self.heap.pop().expect("the heap is non-empty");
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.sift_down(keys, 0);
        }
        Some(top)
    }

    // ARITH: `node` is a valid index and both children are computed from it
    // and bounds-checked before use, so neither the doubling nor the `+ 1`
    // can address outside the heap.
    #[allow(clippy::arithmetic_side_effects)]
    fn sift_down(&mut self, keys: &[Vec<u8>], from: usize) {
        let mut node = from;
        loop {
            let (left, right) = (2 * node + 1, 2 * node + 2);
            let mut smallest = node;
            if left < self.heap.len() && Self::less(keys, self.heap[left], self.heap[smallest]) {
                smallest = left;
            }
            if right < self.heap.len() && Self::less(keys, self.heap[right], self.heap[smallest]) {
                smallest = right;
            }
            if smallest == node {
                return;
            }
            self.heap.swap(node, smallest);
            node = smallest;
        }
    }
}

/// `org.apache.lucene.index.OrdinalMap`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrdinalMap {
    /// `segmentToGlobalOrds[segment][segmentOrd] == globalOrd`.
    segment_to_global: Vec<Vec<i64>>,
    /// `OrdinalMap.valueCount` — the number of distinct terms across every
    /// segment, i.e. the size of the global dictionary.
    value_count: i64,
    /// `OrdinalMap.firstSegments` — for each global ordinal, the first
    /// segment that contains it. See the module doc for how "first" differs
    /// from Java's.
    first_segments: Vec<u32>,
    /// The segment ordinal that global ordinal holds in
    /// [`Self::first_segments`]'s segment. Java derives this from
    /// `globalOrdDeltas`; the same number, stored rather than reconstructed.
    first_segment_ords: Vec<i64>,
}

impl OrdinalMap {
    /// `OrdinalMap.build(owner, TermsEnum[], weights, acceptableOverheadRatio)`
    /// over already-materialized term lists.
    ///
    /// `segments[i]` is segment `i`'s complete terms list **in ordinal order**
    /// — which for a SORTED_SET / SORTED doc-values dictionary is byte-wise
    /// ascending order, exactly what
    /// [`lucene_codecs::terms_dict::decode_all_terms`] returns. A segment with
    /// no values for the field contributes an empty list, the way Java's
    /// `DocValues.emptySortedSet()` sub does.
    ///
    /// Prefer [`OrdinalMap::build_streaming`] where the terms can be
    /// enumerated instead of materialized: this entry point holds every
    /// segment's dictionary, which is ~5x the map it produces (see the module
    /// doc's table). It is the right call only when the caller needs those
    /// lists anyway.
    ///
    /// # Panics
    ///
    /// Debug-asserts that each segment's terms are strictly ascending. Java
    /// makes the same assumption silently (`TermsEnum.next()` is ordered by
    /// contract) and produces a wrong map, not an error, if it is violated.
    pub fn build<T: AsRef<[u8]>>(segments: &[Vec<T>]) -> Self {
        #[cfg(debug_assertions)]
        for (i, terms) in segments.iter().enumerate() {
            for w in terms.windows(2) {
                debug_assert!(
                    w[0].as_ref() < w[1].as_ref(),
                    "segment {i}'s terms are not strictly ascending: \
                     OrdinalMap's merge assumes ordinal order == byte order"
                );
            }
        }
        let mut cursors: Vec<SliceCursor<'_, T>> = segments
            .iter()
            .map(|terms| SliceCursor { terms, next: 0 })
            .collect();
        let mut refs: Vec<&mut dyn TermCursor> = cursors
            .iter_mut()
            .map(|c| c as &mut dyn TermCursor)
            .collect();
        Self::build_streaming(&mut refs).expect("a slice cursor cannot fail")
    }

    /// `OrdinalMap.build(owner, TermsEnum[], weights, acceptableOverheadRatio)`
    /// as Java actually spells it: over **enumerators**, holding no
    /// dictionary.
    ///
    /// `cursors[i]` yields segment `i`'s terms in ascending byte order, and
    /// the term's ordinal in that segment is its position in the enumeration —
    /// Java reads it from `TermsEnum.ord()`, which for a doc-values terms
    /// dictionary is exactly the call count, and taking it from the count
    /// removes an accessor a `TermsCursor` would otherwise have to carry.
    ///
    /// What this holds at once is one reused term buffer per segment plus the
    /// map, where [`OrdinalMap::build`] additionally holds every segment's
    /// whole dictionary — **267 MB of a 319 MB peak** on the 5-segment x
    /// 1 M-term shape the module doc tabulates.
    ///
    /// Errors are the cursors' own (a corrupt terms dictionary); the merge
    /// itself cannot fail.
    pub fn build_streaming(cursors: &mut [&mut dyn TermCursor]) -> lucene_store::Result<Self> {
        let n = cursors.len();
        let mut segment_to_global: Vec<Vec<i64>> = vec![Vec::new(); n];
        // One reused buffer per segment: the cursor's own term is a borrow
        // that dies at its next call, so the merge keeps its own copy —
        // `TermsEnumIndex`'s `BytesRefBuilder`, one per sub.
        let mut current: Vec<Vec<u8>> = vec![Vec::new(); n];
        let mut queue = MergeQueue::with_capacity(n);
        for (segment, cursor) in cursors.iter_mut().enumerate() {
            if let Some(term) = cursor.next_term()? {
                current[segment].extend_from_slice(term);
                queue.push(&current, segment);
            }
        }

        let mut value_count: i64 = 0;
        let mut first_segments: Vec<u32> = Vec::new();
        let mut first_segment_ords: Vec<i64> = Vec::new();
        // `TermsEnumIndex.TermState topState`: the one term the outer loop is
        // assigning a global ordinal to, copied once per *distinct* term
        // rather than once per segment holding it.
        let mut top_term: Vec<u8> = Vec::new();

        while let Some(top) = queue.peek() {
            top_term.clear();
            top_term.extend_from_slice(&current[top]);
            let global_ord = value_count;
            // ARITH: one increment per distinct term, and a term costs at
            // least one byte in some segment's dictionary, so this cannot
            // reach `i64::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                value_count += 1;
            }
            let mut first_segment = usize::MAX;
            let mut first_segment_ord = 0i64;

            // Java's inner `while (true)`: drain every segment standing on
            // this term, recording its ord delta, before moving on.
            loop {
                let segment = queue.peek().expect("the queue was non-empty");
                if segment < first_segment {
                    first_segment = segment;
                    first_segment_ord = segment_to_global[segment].len() as i64;
                }
                // The segment ordinal is the push position: a doc-values terms
                // dictionary numbers its terms densely from zero in
                // enumeration order.
                segment_to_global[segment].push(global_ord);
                if let Some(term) = cursors[segment].next_term()? {
                    debug_assert!(
                        term > &current[segment][..],
                        "segment {segment}'s terms are not strictly ascending: \
                         OrdinalMap's merge assumes ordinal order == byte order"
                    );
                    current[segment].clear();
                    current[segment].extend_from_slice(term);
                    // The key at the top changed in place, so this is Java's
                    // `queue.updateTop()`, not a pop plus a push.
                    queue.update_top(&current);
                } else {
                    queue.pop(&current);
                }
                match queue.peek() {
                    Some(next) if current[next] == top_term => continue,
                    _ => break,
                }
            }
            first_segments.push(first_segment as u32);
            first_segment_ords.push(first_segment_ord);
        }

        Ok(OrdinalMap {
            segment_to_global,
            value_count,
            first_segments,
            first_segment_ords,
        })
    }

    /// `OrdinalMap.getValueCount()` — the number of distinct terms across
    /// every segment.
    pub fn value_count(&self) -> i64 {
        self.value_count
    }

    /// `OrdinalMap.getGlobalOrds(segmentIndex).get(segmentOrd)`.
    ///
    /// Returns `None` for a segment index or segment ordinal this map was not
    /// built with, rather than Java's `ArrayIndexOutOfBoundsException`.
    pub fn global_ord(&self, segment: usize, segment_ord: i64) -> Option<i64> {
        let ords = self.segment_to_global.get(segment)?;
        usize::try_from(segment_ord)
            .ok()
            .and_then(|i| ords.get(i))
            .copied()
    }

    /// `OrdinalMap.getGlobalOrds(segmentIndex)` — the whole segment's map, for
    /// a caller remapping every ordinal of a segment in one pass (which is
    /// what [`crate::facets::merge_segment_counts`] does).
    pub fn segment_ords(&self, segment: usize) -> Option<&[i64]> {
        self.segment_to_global.get(segment).map(Vec::as_slice)
    }

    /// The number of segments this map was built over.
    pub fn segment_count(&self) -> usize {
        self.segment_to_global.len()
    }

    /// `OrdinalMap.getFirstSegmentNumber(globalOrd)` — the first segment
    /// containing this global ordinal's term. "First" is by **input segment
    /// index** here rather than by Java's internal weight-sorted order; see
    /// the module doc.
    pub fn first_segment(&self, global_ord: i64) -> Option<usize> {
        usize::try_from(global_ord)
            .ok()
            .and_then(|i| self.first_segments.get(i))
            .map(|s| *s as usize)
    }

    /// `OrdinalMap.getFirstSegmentOrd(globalOrd)` — that segment's own
    /// ordinal for the term. Same "first" caveat as [`Self::first_segment`].
    pub fn first_segment_ord(&self, global_ord: i64) -> Option<i64> {
        usize::try_from(global_ord)
            .ok()
            .and_then(|i| self.first_segment_ords.get(i))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(v: &[&str]) -> Vec<Vec<u8>> {
        v.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    /// The exact worked example in `OrdinalMap`'s own class javadoc:
    /// global `bar -> 0, cat -> 1, dog -> 2, foo -> 3`; segment 0 holds
    /// `bar, foo`, segment 1 holds `cat, dog`.
    #[test]
    fn javadoc_worked_example() {
        let map = OrdinalMap::build(&[terms(&["bar", "foo"]), terms(&["cat", "dog"])]);
        assert_eq!(map.value_count(), 4);
        assert_eq!(map.segment_ords(0), Some(&[0i64, 3][..]));
        assert_eq!(map.segment_ords(1), Some(&[1i64, 2][..]));
    }

    #[test]
    fn a_term_shared_by_every_segment_gets_one_global_ordinal() {
        let map = OrdinalMap::build(&[
            terms(&["a", "b"]),
            terms(&["a", "c"]),
            terms(&["a", "b", "c"]),
        ]);
        assert_eq!(map.value_count(), 3);
        assert_eq!(map.segment_ords(0), Some(&[0i64, 1][..]));
        assert_eq!(map.segment_ords(1), Some(&[0i64, 2][..]));
        assert_eq!(map.segment_ords(2), Some(&[0i64, 1, 2][..]));
        // "first segment" ties break to the lowest input index.
        assert_eq!(map.first_segment(0), Some(0));
        assert_eq!(map.first_segment(2), Some(1));
        assert_eq!(map.first_segment_ord(2), Some(1));
    }

    #[test]
    fn an_empty_segment_contributes_nothing_and_is_still_addressable() {
        let map = OrdinalMap::build(&[terms(&["a"]), Vec::<Vec<u8>>::new(), terms(&["b"])]);
        assert_eq!(map.value_count(), 2);
        assert_eq!(map.segment_count(), 3);
        assert_eq!(map.segment_ords(1), Some(&[][..]));
        assert_eq!(map.global_ord(1, 0), None);
        assert_eq!(map.global_ord(2, 0), Some(1));
    }

    #[test]
    fn no_segments_at_all_is_an_empty_map_not_a_panic() {
        let map = OrdinalMap::build::<Vec<u8>>(&[]);
        assert_eq!(map.value_count(), 0);
        assert_eq!(map.segment_count(), 0);
        assert_eq!(map.global_ord(0, 0), None);
        assert_eq!(map.first_segment(0), None);
        assert_eq!(map.first_segment_ord(0), None);
    }

    #[test]
    fn out_of_range_lookups_return_none_rather_than_panicking() {
        let map = OrdinalMap::build(&[terms(&["a", "b"])]);
        assert_eq!(map.global_ord(0, 1), Some(1));
        assert_eq!(map.global_ord(0, 2), None);
        assert_eq!(map.global_ord(0, -1), None);
        assert_eq!(map.global_ord(9, 0), None);
        assert_eq!(map.segment_ords(9), None);
        assert_eq!(map.first_segment(-1), None);
        assert_eq!(map.first_segment_ord(-1), None);
        assert_eq!(map.first_segment(99), None);
        assert_eq!(map.first_segment_ord(99), None);
    }

    /// The global order is byte order, not the order segments were merged in:
    /// a term that sorts first but appears only in the *last* segment still
    /// takes global ordinal 0.
    #[test]
    fn global_ordinals_follow_byte_order_across_segments() {
        let map = OrdinalMap::build(&[terms(&["m", "z"]), terms(&["b", "m"])]);
        assert_eq!(map.segment_ords(0), Some(&[1i64, 2][..]));
        assert_eq!(map.segment_ords(1), Some(&[0i64, 1][..]));
        assert_eq!(map.first_segment(0), Some(1));
    }

    /// Byte order, not `str` order: a term with a byte above 0x7F must not be
    /// compared as UTF-8 code points (Java compares `BytesRef`s).
    #[test]
    fn ordering_is_over_raw_bytes() {
        let a = vec![vec![0xC3u8, 0xA9]]; // "é"
        let b = vec![vec![0x7Au8]]; // "z"
        let map = OrdinalMap::build(&[a, b]);
        assert_eq!(map.segment_ords(1), Some(&[0i64][..]), "0x7A < 0xC3");
        assert_eq!(map.segment_ords(0), Some(&[1i64][..]));
    }

    /// A [`TermCursor`] over an owned list, so the streaming entry point can
    /// be driven from a test without a `.dvd`. Deliberately hands back a
    /// **reused** buffer, which is the contract a real
    /// `terms_dict::TermsCursor` relies on: a merge that assumed the previous
    /// term stayed valid would read the wrong bytes here.
    struct VecCursor {
        terms: Vec<Vec<u8>>,
        next: usize,
        buffer: Vec<u8>,
        fail_at: Option<usize>,
    }

    impl VecCursor {
        fn new(v: &[&str]) -> Self {
            VecCursor {
                terms: terms(v),
                next: 0,
                buffer: Vec::new(),
                fail_at: None,
            }
        }
    }

    impl TermCursor for VecCursor {
        fn next_term(&mut self) -> lucene_store::Result<Option<&[u8]>> {
            if self.fail_at == Some(self.next) {
                return Err(lucene_store::Error::Corrupted("bad block".into()));
            }
            let Some(term) = self.terms.get(self.next) else {
                return Ok(None);
            };
            self.next += 1;
            self.buffer.clear();
            self.buffer.extend_from_slice(term);
            Ok(Some(&self.buffer))
        }
    }

    fn streamed(mut cursors: Vec<VecCursor>) -> lucene_store::Result<OrdinalMap> {
        let mut refs: Vec<&mut dyn TermCursor> = cursors
            .iter_mut()
            .map(|c| c as &mut dyn TermCursor)
            .collect();
        OrdinalMap::build_streaming(&mut refs)
    }

    /// The two entry points are one algorithm, so they must agree exactly --
    /// on every field of the map, not just `valueCount`. Eight segments, so
    /// the heap is deep enough that both children of a node are compared on
    /// the way down (a two- or three-segment case never reaches the right
    /// child).
    #[test]
    fn build_streaming_agrees_with_build_over_the_same_terms() {
        let shapes: [&[&[&str]]; 4] = [
            &[&["bar", "foo"], &["cat", "dog"]],
            &[&["a", "b"], &["a", "c"], &["a", "b", "c"]],
            &[&["a"], &[], &["b"]],
            &[
                &["h", "p"],
                &["a", "z"],
                &["c", "h", "q"],
                &["b"],
                &["d", "e", "f"],
                &["g", "h"],
                &["m"],
                &["a", "n", "z"],
            ],
        ];
        for shape in shapes {
            let materialized: Vec<Vec<Vec<u8>>> = shape.iter().map(|s| terms(s)).collect();
            let expected = OrdinalMap::build(&materialized);
            let got = streamed(shape.iter().map(|s| VecCursor::new(s)).collect()).unwrap();
            assert_eq!(got, expected, "shape {shape:?}");
        }
    }

    /// The eight-segment shape spelled out, so the agreement test above
    /// cannot pass by both halves being wrong in the same way.
    #[test]
    fn build_streaming_assigns_global_ordinals_in_byte_order() {
        let map = streamed(vec![
            VecCursor::new(&["c", "e"]),
            VecCursor::new(&["a", "e"]),
            VecCursor::new(&["b", "d"]),
        ])
        .unwrap();
        // Global: a=0, b=1, c=2, d=3, e=4.
        assert_eq!(map.value_count(), 5);
        assert_eq!(map.segment_ords(0), Some(&[2i64, 4][..]));
        assert_eq!(map.segment_ords(1), Some(&[0i64, 4][..]));
        assert_eq!(map.segment_ords(2), Some(&[1i64, 3][..]));
        // "e" is in segments 0 and 1; "first" is the lowest input index, and
        // its ordinal there is 1.
        assert_eq!(map.first_segment(4), Some(0));
        assert_eq!(map.first_segment_ord(4), Some(1));
    }

    /// A cursor's error is the caller's, not a panic and not a silently short
    /// map -- the failure mode a corrupt `.dvd` produces.
    #[test]
    fn a_cursor_error_propagates_out_of_build_streaming() {
        let mut ok = VecCursor::new(&["a", "b"]);
        let mut bad = VecCursor::new(&["a", "c"]);
        bad.fail_at = Some(1);
        let mut refs: Vec<&mut dyn TermCursor> = vec![
            &mut ok as &mut dyn TermCursor,
            &mut bad as &mut dyn TermCursor,
        ];
        assert!(matches!(
            OrdinalMap::build_streaming(&mut refs),
            Err(lucene_store::Error::Corrupted(_))
        ));
    }

    /// The very first `next_term` failing is a separate branch from one
    /// failing mid-merge (the priming loop, not the drain loop).
    #[test]
    fn a_cursor_that_fails_on_its_first_term_propagates_too() {
        let mut bad = VecCursor::new(&["a"]);
        bad.fail_at = Some(0);
        let mut refs: Vec<&mut dyn TermCursor> = vec![&mut bad as &mut dyn TermCursor];
        assert!(matches!(
            OrdinalMap::build_streaming(&mut refs),
            Err(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn build_streaming_over_no_cursors_is_an_empty_map() {
        let map = streamed(Vec::new()).unwrap();
        assert_eq!(map.value_count(), 0);
        assert_eq!(map.segment_count(), 0);
    }

    /// The streaming entry point carries the same debug-time guard as the
    /// materialized one, and it has to be its own check: `build`'s runs over
    /// the lists before the merge starts, and a cursor has no list to scan.
    #[test]
    #[should_panic(expected = "not strictly ascending")]
    fn an_unsorted_cursor_is_caught_in_debug_builds() {
        let _ = streamed(vec![VecCursor::new(&["b", "a"])]);
    }

    #[test]
    #[should_panic(expected = "not strictly ascending")]
    fn unsorted_input_is_caught_in_debug_builds() {
        OrdinalMap::build(&[terms(&["b", "a"])]);
    }
}
