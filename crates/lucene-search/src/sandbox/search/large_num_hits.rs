//! `LargeNumHitsTopDocsCollector` and its manager: the top hits by score
//! for a large `numHits`, kept in a plain list until there are more than
//! `numHits` of them, and only then heapified (`HitQueue`, not prefilled).
//!
//! Collecting is `ScoreMode.COMPLETE`: no threshold is handed to the scorer,
//! and the total is exact. Past `numHits` hits a document replaces the
//! queue's weakest -- lowest score, then highest document -- only when it
//! scores strictly higher. `topDocs(howMany)` reads `min(howMany,
//! totalHits)` hits: from the list, sorted by score descending then document
//! ascending; or, once the queue exists, by popping it -- the queue's
//! *weakest* `howMany`, strongest first, which is its best only when
//! `howMany` is the queue's size (as `topDocs()` asks), and which leaves the
//! popped hits gone for a later call. `topDocs(0)` is Java's
//! `EMPTY_TOPDOCS`, a total of `0`. The manager merges its collectors'
//! `topDocs()` with `TopDocs.merge(0, numHits, ...)`.
//!
//! Differences: the documents a collector sees are already global (a search
//! shifts them by the segment's base, Java's `doc + docBase`); the queue is a
//! [`BinaryHeap`] under `HitQueue.lessThan`'s order (no two hits share a
//! document, so its pops are `HitQueue`'s); a queue popped dry yields fewer
//! hits, where Java's array leads with `null`s; `toString` is not ported.
//! Scores are non-negative and never `NaN`, as Java's collector asserts.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::collector::{ScoreMode, ScoringCollector, TotalHits, TotalHitsRelation};
use crate::collectors::CollectorManager;
use crate::top_docs::{merge, ShardScoreDoc, TieBreaker, TopDocs};
use crate::{Error, Result};

/// A hit in the queue, ordered so that the max-heap's top is `HitQueue`'s
/// least: the lowest score, then the highest document.
#[derive(Debug, Clone, Copy)]
struct Weakest(ShardScoreDoc);

impl PartialEq for Weakest {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl Eq for Weakest {}

impl PartialOrd for Weakest {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Weakest {
    /// `HitQueue.lessThan(a, b)`: equal scores (`==`, so `-0.0` ties `0.0`)
    /// order by document, the higher one less; otherwise the lower score is
    /// less. Reversed here, so that "less" is greater.
    fn cmp(&self, o: &Self) -> Ordering {
        let (a, b) = (self.0, o.0);
        if a.score == b.score {
            a.doc.cmp(&b.doc)
        } else if a.score < b.score {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

/// `LargeNumHitsTopDocsCollector(requestedHitCount)`.
#[derive(Debug, Clone)]
pub struct LargeNumHitsTopDocsCollector {
    requested_hit_count: usize,
    hits: Vec<ShardScoreDoc>,
    pq: Option<BinaryHeap<Weakest>>,
    total_hits: usize,
}

impl LargeNumHitsTopDocsCollector {
    /// `LargeNumHitsTopDocsCollector(requestedHitCount)`.
    ///
    /// # Errors
    /// A `requested_hit_count` of `0` or less (`"requestedHitCount must be >
    /// 0"`).
    pub fn new(requested_hit_count: i32) -> Result<Self> {
        let requested_hit_count = usize::try_from(requested_hit_count)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::IllegalArgument("requestedHitCount must be > 0".into()))?;
        Ok(Self {
            requested_hit_count,
            hits: Vec::new(),
            pq: None,
            total_hits: 0,
        })
    }

    /// `totalHits`: every document collected.
    pub fn total_hits(&self) -> usize {
        self.total_hits
    }

    /// `topDocs(howMany)`.
    ///
    /// # Errors
    /// A negative `how_many` (`"Number of hits requested must not be
    /// negative but value was ..."`).
    pub fn top_docs_n(&mut self, how_many: i32) -> Result<TopDocs> {
        let how_many = usize::try_from(how_many).map_err(|_| {
            Error::IllegalArgument(format!(
                "Number of hits requested must not be negative but value was {how_many}"
            ))
        })?;
        if how_many == 0 {
            return Ok(TopDocs {
                total_hits: TotalHits {
                    value: 0,
                    relation: TotalHitsRelation::EqualTo,
                },
                score_docs: Vec::new(),
            });
        }
        let how_many = how_many.min(self.total_hits);
        let score_docs = self.populate_results(how_many);
        Ok(TopDocs {
            total_hits: TotalHits {
                value: self.total_hits as u64,
                relation: TotalHitsRelation::EqualTo,
            },
            score_docs,
        })
    }

    /// `populateResults(results, howMany)`.
    fn populate_results(&mut self, how_many: usize) -> Vec<ShardScoreDoc> {
        if let Some(pq) = &mut self.pq {
            // `results[i] = pq.pop()` for `i` from `howMany - 1` down to `0`.
            let mut out: Vec<ShardScoreDoc> = std::iter::from_fn(|| pq.pop().map(|w| w.0))
                .take(how_many)
                .collect();
            out.reverse();
            return out;
        }
        // `Comparator.comparing(score).reversed().thenComparing(doc)`, over
        // boxed `Float`s: `Float.compare`'s total order.
        self.hits
            .sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
        self.hits.iter().take(how_many).copied().collect()
    }

    /// `topDocs()`: `topDocs(min(totalHits, requestedHitCount))`.
    pub fn top_docs(&mut self) -> TopDocs {
        let n = self.total_hits.min(self.requested_hit_count);
        // `n` is at most `requested_hit_count`, which `new` took from an
        // `i32`, so it converts back and is not negative.
        let n = i32::try_from(n).unwrap_or(i32::MAX);
        self.top_docs_n(n)
            .unwrap_or_else(|_| unreachable!("a non-negative count"))
    }
}

impl ScoringCollector for LargeNumHitsTopDocsCollector {
    /// `getLeafCollector(context).collect(doc)`.
    fn collect(&mut self, doc_id: i32, score: f32) {
        if self.total_hits < self.requested_hit_count {
            self.hits.push(ShardScoreDoc::new(doc_id, score));
            self.total_hits = self.total_hits.saturating_add(1);
            return;
        }
        // At `totalHits == requestedHitCount`: the list becomes the queue.
        let hits = &mut self.hits;
        let pq = self
            .pq
            .get_or_insert_with(|| std::mem::take(hits).into_iter().map(Weakest).collect());
        if let Some(mut top) = pq.peek_mut() {
            if score > top.0.score {
                // `pqTop.doc = ...; pqTop.score = ...; pq.updateTop()`.
                *top = Weakest(ShardScoreDoc::new(doc_id, score));
            }
        }
        self.total_hits = self.total_hits.saturating_add(1);
    }

    fn score_mode(&self) -> ScoreMode {
        ScoreMode::Complete
    }
}

/// `LargeNumHitsTopDocsCollectorManager(numHits)`.
#[derive(Debug, Clone, Copy)]
pub struct LargeNumHitsTopDocsCollectorManager {
    num_hits: i32,
}

impl LargeNumHitsTopDocsCollectorManager {
    /// `LargeNumHitsTopDocsCollectorManager(numHits)`.
    ///
    /// # Errors
    /// A `num_hits` of `0` or less (`"numHits must be > 0, got ..."`).
    pub fn new(num_hits: i32) -> Result<Self> {
        if num_hits <= 0 {
            return Err(Error::IllegalArgument(format!(
                "numHits must be > 0, got {num_hits}"
            )));
        }
        Ok(Self { num_hits })
    }
}

impl CollectorManager for LargeNumHitsTopDocsCollectorManager {
    type Collector = LargeNumHitsTopDocsCollector;
    type Output = TopDocs;

    fn new_collector(&self) -> Result<Self::Collector> {
        LargeNumHitsTopDocsCollector::new(self.num_hits)
    }

    /// `TopDocs.merge(0, numHits, collectors' topDocs())`.
    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<TopDocs> {
        let tops: Vec<TopDocs> = collectors.into_iter().map(|mut c| c.top_docs()).collect();
        // `num_hits` is positive (`new`).
        let top_n = usize::try_from(self.num_hits).unwrap_or(0);
        merge(0, top_n, &tops, &TieBreaker::Default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs(t: &TopDocs) -> Vec<(i32, f32)> {
        t.score_docs.iter().map(|s| (s.doc, s.score)).collect()
    }

    fn collected(n: i32, hits: &[(i32, f32)]) -> LargeNumHitsTopDocsCollector {
        let mut c = LargeNumHitsTopDocsCollector::new(n).unwrap();
        for &(d, s) in hits {
            c.collect(d, s);
        }
        c
    }

    #[test]
    fn rejects_what_java_rejects() {
        for n in [0, -3] {
            assert_eq!(
                LargeNumHitsTopDocsCollector::new(n)
                    .unwrap_err()
                    .to_string(),
                Error::IllegalArgument("requestedHitCount must be > 0".into()).to_string()
            );
        }
        assert!(LargeNumHitsTopDocsCollectorManager::new(0)
            .unwrap_err()
            .to_string()
            .contains("numHits must be > 0, got 0"));
        let mut c = collected(2, &[(0, 1.0)]);
        assert!(c
            .top_docs_n(-1)
            .unwrap_err()
            .to_string()
            .contains("must not be negative but value was -1"));
    }

    #[test]
    fn below_the_count_sorts_the_list() {
        let mut c = collected(5, &[(3, 1.0), (1, 2.0), (2, 1.0)]);
        assert_eq!(c.score_mode(), ScoreMode::Complete);
        assert_eq!(c.total_hits(), 3);
        let t = c.top_docs();
        assert_eq!(docs(&t), vec![(1, 2.0), (2, 1.0), (3, 1.0)]);
        assert_eq!(t.total_hits.value, 3);
        assert_eq!(c.top_docs_n(0).unwrap().total_hits.value, 0);
        assert_eq!(docs(&c.top_docs_n(9).unwrap()).len(), 3);
    }

    #[test]
    fn past_the_count_keeps_the_strictly_better() {
        // Ties with the weakest lose: document 9 at 1.0 stays out.
        let hits = [(0, 1.0), (1, 3.0), (2, 2.0), (9, 1.0), (10, 5.0), (11, 0.5)];
        let mut c = collected(3, &hits);
        assert_eq!(c.total_hits(), 6);
        assert_eq!(docs(&c.top_docs()), vec![(10, 5.0), (1, 3.0), (2, 2.0)]);
        // Popped dry: nothing is left for a second call.
        assert!(c.top_docs().score_docs.is_empty());
    }

    #[test]
    fn a_short_read_of_the_queue_is_its_weakest() {
        let mut c = collected(3, &[(0, 1.0), (1, 3.0), (2, 2.0), (3, 4.0)]);
        // The queue holds 3, 2 and 1 (scores 4, 2, 3); two pops are its
        // weakest two, strongest first.
        assert_eq!(docs(&c.top_docs_n(2).unwrap()), vec![(1, 3.0), (2, 2.0)]);
    }

    #[test]
    fn equal_scores_order_by_document() {
        let mut c = collected(2, &[(5, 0.0), (4, -0.0), (7, 0.0)]);
        // `-0.0 == 0.0`: the queue's weakest is the highest document.
        assert_eq!(docs(&c.top_docs()), vec![(4, -0.0), (5, 0.0)]);
    }

    #[test]
    fn the_manager_merges_its_collectors() {
        let m = LargeNumHitsTopDocsCollectorManager::new(2).unwrap();
        let mut a = m.new_collector().unwrap();
        a.collect(0, 1.0);
        a.collect(1, 4.0);
        let mut b = m.new_collector().unwrap();
        b.collect(5, 3.0);
        let t = m.reduce(vec![a, b]).unwrap();
        assert_eq!(docs(&t), vec![(1, 4.0), (5, 3.0)]);
        assert_eq!(t.total_hits.value, 3);
    }
}
