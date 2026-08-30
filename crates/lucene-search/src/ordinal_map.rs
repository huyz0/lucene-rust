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
//! - **The input is materialized, where Java's streams.** `OrdinalMap.build`
//!   takes `TermsEnum[]` and never holds a dictionary; [`OrdinalMap::build`]
//!   takes every segment's complete term list. It is what the port has to work
//!   with: this crate's SORTED_SET dictionaries are read whole by
//!   [`lucene_codecs::terms_dict::decode_all_terms`], and a streaming
//!   `TermsEnum`-shaped input would need a cursor API over the doc-values
//!   terms dictionary that does not exist yet.
//!
//!   **Measured** (`examples/ordinal_map_memory.rs`, 17-byte terms, Linux
//!   RSS with the allocation totals agreeing to within the allocator's own
//!   slack):
//!
//!   | shape | materialized input | the map itself | peak |
//!   |---|---|---|---|
//!   | 5 segments x 1 M terms, 1.2 M global | 267 MB | 52 MB | 319 MB |
//!   | 10 x 200 k, 380 k global | 107 MB | 20 MB | 127 MB |
//!   | 20 x 50 k, 161 k global | 53 MB | 10 MB | 63 MB |
//!
//!   So the input is **~5x the map** and ~84% of the peak: this divergence
//!   costs materially more than the `Vec<i64>` representation above, and is
//!   the one worth closing first. Closing it is not a change to this file --
//!   it needs a `TermsEnum`-shaped cursor over a doc-values terms dictionary
//!   in `lucene-codecs`, and every caller (`facets`) to stop calling
//!   `decode_all_terms`. Recorded in `docs/sweep/m2/c29-search-carryovers.md`
//!   with the handoff.
//! - **No `IndexReader.CacheKey` owner.** Java carries one so
//!   `DefaultSortedSetDocValuesReaderState` can cache the map against the
//!   reader it was built for. This port has no reader cache-key mechanism, so
//!   the caller owns the lifetime.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One segment's position in the k-way merge: the segment index, and how far
/// through its (already sorted, already unique) term list we are.
///
/// `Ord` is reversed on the term and then on the segment index so that
/// [`BinaryHeap`] — a max-heap — pops the **smallest** term first, and ties
/// break towards the lowest segment index. That tie rule is what makes
/// [`OrdinalMap::first_segment`] well-defined.
struct Cursor<'a> {
    term: &'a [u8],
    segment: usize,
    next: usize,
}

impl Ord for Cursor<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .term
            .cmp(self.term)
            .then_with(|| other.segment.cmp(&self.segment))
    }
}
impl PartialOrd for Cursor<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Cursor<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Cursor<'_> {}

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
    /// `OrdinalMap.build(owner, TermsEnum[], weights, acceptableOverheadRatio)`.
    ///
    /// `segments[i]` is segment `i`'s complete terms list **in ordinal order**
    /// — which for a SORTED_SET / SORTED doc-values dictionary is byte-wise
    /// ascending order, exactly what
    /// [`lucene_codecs::terms_dict::decode_all_terms`] returns. A segment with
    /// no values for the field contributes an empty list, the way Java's
    /// `DocValues.emptySortedSet()` sub does.
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

        let mut segment_to_global: Vec<Vec<i64>> =
            segments.iter().map(|s| vec![0i64; s.len()]).collect();
        let mut heap: BinaryHeap<Cursor<'_>> = BinaryHeap::with_capacity(segments.len());
        for (segment, terms) in segments.iter().enumerate() {
            if let Some(first) = terms.first() {
                heap.push(Cursor {
                    term: first.as_ref(),
                    segment,
                    next: 1,
                });
            }
        }

        let mut value_count: i64 = 0;
        let mut first_segments: Vec<u32> = Vec::new();
        let mut first_segment_ords: Vec<i64> = Vec::new();
        // Borrowed from `segments`, not copied: the previous pop's term is
        // still alive in the caller's own dictionary, so keeping a `Vec<u8>`
        // here was one heap allocation and one copy **per distinct term** --
        // 1.2 M of them on the 5-segment x 1 M-term shape measured in
        // `docs/sweep/m2/c29-search-carryovers.md`, for a value that is only
        // ever compared and then dropped.
        let mut current: Option<&[u8]> = None;

        while let Some(cursor) = heap.pop() {
            // A new global ordinal starts whenever the popped term differs
            // from the one the previous pop assigned -- the heap's ordering
            // guarantees every segment holding a term pops consecutively.
            let is_new = match current {
                Some(prev) => prev != cursor.term,
                None => true,
            };
            if is_new {
                current = Some(cursor.term);
                value_count += 1;
                first_segments.push(cursor.segment as u32);
                first_segment_ords.push(cursor.next as i64 - 1);
            }
            segment_to_global[cursor.segment][cursor.next - 1] = value_count - 1;

            if let Some(term) = segments[cursor.segment].get(cursor.next) {
                heap.push(Cursor {
                    term: term.as_ref(),
                    segment: cursor.segment,
                    next: cursor.next + 1,
                });
            }
        }

        OrdinalMap {
            segment_to_global,
            value_count,
            first_segments,
            first_segment_ords,
        }
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

    #[test]
    #[should_panic(expected = "not strictly ascending")]
    fn unsorted_input_is_caught_in_debug_builds() {
        OrdinalMap::build(&[terms(&["b", "a"])]);
    }
}
