//! `LuceneTestCase.newSearcher(reader)`: a searcher over the test's reader,
//! as a random mix of the configurations Lucene ships -- concurrent slices
//! or not, a never-expiring timeout or none -- that always runs its
//! searches under the asserting wrappers (`AssertingIndexSearcher`).
//!
//! The draws follow Lucene 10.5.0's `newSearcher(r)` (`maybeWrap` and
//! `wrapWithAssertions` both `true`, `useThreads` a coin) call for call,
//! from the test's `random()`; [`NewSearcher::draw`] makes them,
//! [`NewSearcher::configure`] applies them to a searcher, and
//! [`NewSearcher::searching`] runs one search under the asserting
//! wrappers, as `AssertingIndexSearcher` runs each.
//!
//! Differs:
//!
//! - `maybeWrapReader`'s `rarely()` is drawn, but a reader it would wrap
//!   (`wrapReader`: slow, parallel or filter readers) is searched as it is;
//!   [`NewSearcher::wrap_reader`] records the draw. `wrapReader`'s own
//!   draws are not made.
//! - `checkReader` (one time in 500, for a leaf reader) is drawn and not
//!   run: the reader is always a `DirectoryReader`.
//! - `IndexSearcher(reader)` and `IndexSearcher(reader.getContext())` are
//!   the same searcher here; the coin is drawn.
//! - The slices are drawn when the searcher is made (Java computes them,
//!   and draws `frequently()`, on the first search), and a slice is always
//!   whole segments: the port's slices do not partition a segment, so the
//!   `frequently()` that would allow it is drawn and recorded only. The
//!   executor coin is drawn; slices run on scoped threads either way.
//! - The class environment's similarity is not set (the searcher keeps
//!   the default, or the caller's), nor `MAYBE_CACHE_POLICY` (the port's
//!   searcher has no query cache of its own).
//! - `AssertingIndexSearcher`'s wrappers are installed around each search
//!   ([`lucene_search::test_support::asserting`], seeded from its random as
//!   [`super::asserting::AssertingIndexSearcher`] seeds them), on the
//!   calling thread; `AssertingCollector` wraps only searches through
//!   [`super::asserting::AssertingIndexSearcher`].

use std::cell::RefCell;
use std::sync::Arc;

use lucene_search::index_searcher::IndexSearcher;
use lucene_search::reader::exitable::QueryTimeout;
use lucene_search::test_support;
use lucene_util::java_random::JavaRandom;

use crate::util::lucene_test_case;

/// `IndexSearcher.slices(leaves, maxDocsPerSlice, maxSegmentsPerSlice,
/// false)`: segments (by their `maxDoc`, in leaf order) grouped into
/// slices, largest first; a segment above `max_docs_per_slice` is a slice
/// of its own, and a group closes once it holds `max_segments_per_slice`
/// segments or more than `max_docs_per_slice` documents. Each slice lists
/// its segments in leaf order (`LeafSlice` sorts by doc base).
#[must_use]
pub fn leaf_slices(
    max_docs: &[i32],
    max_docs_per_slice: i32,
    max_segments_per_slice: i32,
) -> Vec<Vec<usize>> {
    let mut sorted: Vec<usize> = (0..max_docs.len()).collect();
    // `Collections.reverseOrder(comparingInt(maxDoc))` over a stable sort.
    sorted.sort_by(|&a, &b| max_docs[b].cmp(&max_docs[a]));
    let mut grouped: Vec<Vec<usize>> = Vec::new();
    let mut doc_sum: i64 = 0;
    let mut open = false;
    for leaf in sorted {
        if max_docs[leaf] > max_docs_per_slice {
            debug_assert!(!open);
            grouped.push(vec![leaf]);
        } else {
            if open {
                grouped.last_mut().expect("an open group").push(leaf);
            } else {
                grouped.push(vec![leaf]);
                open = true;
            }
            doc_sum += i64::from(max_docs[leaf]);
            let size = grouped.last().map_or(0, Vec::len);
            if size >= max_segments_per_slice as usize || doc_sum > i64::from(max_docs_per_slice) {
                open = false;
                doc_sum = 0;
            }
        }
    }
    for slice in &mut grouped {
        slice.sort_unstable();
    }
    grouped
}

/// The concurrent branch's slicing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceDraws {
    pub max_docs_per_slice: i32,
    pub max_segments_per_slice: i32,
    /// The executor coin: `false` is Java's shared executor.
    pub no_executor: bool,
    /// `frequently()` under `INTRA_SEGMENT`: whether Java would partition
    /// segments.
    pub allow_partitions: bool,
}

/// What `newSearcher(r)` drew.
#[derive(Debug, Clone)]
pub struct NewSearcher {
    /// `maybeWrapReader`'s `rarely()` (`NONE` concurrency only).
    pub wrap_reader: bool,
    /// The slicing, for the concurrent branch.
    pub slices: Option<SliceDraws>,
    /// `setTimeout(() -> false)` (concurrent branch only).
    pub timeout: bool,
    /// `AssertingIndexSearcher`'s own random.
    random: RefCell<JavaRandom>,
}

/// `() -> false`: a timeout that never expires.
#[derive(Debug)]
struct NeverTimesOut;

impl QueryTimeout for NeverTimesOut {
    fn should_exit(&self) -> bool {
        false
    }
}

impl NewSearcher {
    /// `newSearcher(r)` under the given `TEST_NIGHTLY` and multiplier,
    /// drawing from the test's `random`.
    pub fn draw(random: &mut JavaRandom, nightly: bool, multiplier: i32) -> Self {
        let use_threads = random.next_boolean();
        if !use_threads {
            // `Concurrency.NONE`.
            let wrap_reader = lucene_test_case::rarely(random, nightly, multiplier);
            let _check_reader = random.next_int_bounded(500) == 0;
            let _from_context = random.next_boolean();
            return Self {
                wrap_reader,
                slices: None,
                timeout: false,
                random: RefCell::new(JavaRandom::new(random.next_long())),
            };
        }
        // `Concurrency.INTRA_SEGMENT`.
        let no_executor = random.next_boolean();
        let max_docs_per_slice = if random.next_boolean() {
            1
        } else {
            1 + random.next_int_bounded(1000)
        };
        let max_segments_per_slice = if random.next_boolean() {
            1
        } else {
            1 + random.next_int_bounded(10)
        };
        let _from_context = random.next_boolean();
        let asserting_seed = random.next_long();
        let timeout = random.next_boolean();
        let allow_partitions = !lucene_test_case::rarely(random, nightly, multiplier);
        Self {
            wrap_reader: false,
            slices: Some(SliceDraws {
                max_docs_per_slice,
                max_segments_per_slice,
                no_executor,
                allow_partitions,
            }),
            timeout,
            random: RefCell::new(JavaRandom::new(asserting_seed)),
        }
    }

    /// Configures `searcher` as drawn: its slices (over its segments'
    /// `maxDoc`s) and timeout.
    ///
    /// # Panics
    /// When the slices do not cover the searcher's segments, which
    /// [`leaf_slices`] never produces.
    pub fn configure(&self, searcher: &mut IndexSearcher<'_, '_>) {
        if let Some(d) = self.slices {
            let max_docs: Vec<i32> = searcher
                .segments()
                .iter()
                .map(|s| s.max_doc.unwrap_or(0))
                .collect();
            let slices = leaf_slices(&max_docs, d.max_docs_per_slice, d.max_segments_per_slice);
            searcher
                .set_slices(slices)
                .expect("slices over every segment");
        }
        if self.timeout {
            searcher.set_timeout(Some(Arc::new(NeverTimesOut)));
        }
    }

    /// Runs one search, `f`, under the asserting wrappers.
    pub fn searching<R>(&self, f: impl FnOnce() -> R) -> R {
        let _asserting = test_support::asserting(self.random.borrow_mut().next_long());
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::test_index::TestIndex;
    use lucene_index::document::{Document, Store, StringField};

    #[test]
    fn both_branches_are_drawn_and_configure_the_searcher() {
        let docs: Vec<Vec<Document>> = (0..3)
            .map(|i| {
                let mut d = Document::new();
                d.add(StringField::new("id", i.to_string(), Store::No));
                vec![d]
            })
            .collect();
        let analyzer = Arc::new(
            crate::analysis::mock_analyzer::MockAnalyzer::new(&mut JavaRandom::new(1)).build(),
        );
        let index = TestIndex::build(analyzer, &docs, &[], &[]);
        let (mut plain, mut sliced, mut timed) = (0, 0, 0);
        let mut random = JavaRandom::new(7);
        for _ in 0..64 {
            let draws = NewSearcher::draw(&mut random, false, 1);
            index.with_segments(|segments, norms| {
                let mut s = IndexSearcher::new(segments, norms).unwrap();
                draws.configure(&mut s);
                match draws.slices {
                    None => {
                        plain += 1;
                        assert_eq!(s.slices(), [vec![0, 1, 2]]);
                    }
                    Some(d) => {
                        sliced += 1;
                        let want =
                            leaf_slices(&[1, 1, 1], d.max_docs_per_slice, d.max_segments_per_slice);
                        assert_eq!(s.slices(), want.as_slice());
                    }
                }
                timed += usize::from(draws.timeout);
                assert_eq!(s.timeout().is_some(), draws.timeout);
                assert!(draws.searching(test_support::asserting_active));
                assert!(!test_support::asserting_active());
            });
        }
        assert!(plain > 0 && sliced > 0 && timed > 0);
    }

    #[test]
    fn a_large_segment_is_a_slice_of_its_own() {
        assert_eq!(
            leaf_slices(&[3, 50, 2, 4], 10, 2),
            [vec![1], vec![0, 3], vec![2]]
        );
        assert_eq!(leaf_slices(&[], 1, 1), Vec::<Vec<usize>>::new());
        assert!(!NeverTimesOut.should_exit());
    }
}
