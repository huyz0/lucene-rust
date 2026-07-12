//! `Collector`/`LeafCollector`-equivalent, pared down to this slice's scope:
//! a plain callback trait invoked once per matching (live) doc ID, in
//! ascending order. Real Lucene's `Collector`/`LeafCollector` split exists to
//! let a collector rebind per-segment state (e.g. a `Scorer` reference) when
//! `IndexSearcher` moves to the next leaf; that split has no work to do yet
//! since this slice never federates more than one segment (see `lib.rs`'s
//! module doc), so a single flat trait stands in for both.
//!
//! `search_term_query` (`lib.rs`) is generic over `C: Collector` rather than
//! taking `&mut dyn Collector`, per the `rust-performance` skill's
//! "monomorphize per-doc loops, `dyn` only at Query/Weight level" rule — the
//! per-doc `collect()` call in the hot loop is a direct (inlinable) call, not
//! a vtable dispatch.
//!
//! ## `ScoringCollector` (task #13's addition) — a new trait, not a breaking
//! change to `Collector`
//!
//! `lib.rs`'s module doc, written when this file only had unscored matching,
//! already flagged that relevance scoring would need `Collector::collect` to
//! grow a `score: f32` parameter, and called that "a breaking signature
//! change, every existing `Collector` impl's signature changes". Having now
//! reached that point, this port takes the **non-breaking path instead**: a
//! separate [`ScoringCollector`] trait with its own `collect(doc_id, score)`
//! method, rather than editing [`Collector`] in place. Reasoning:
//!
//! - **Not every caller needs a score.** `CountCollector`/`VecCollector` (and
//!   `search_term_query`/`search_boolean_query`, which only ever determine
//!   matching, not ranking) have no use for a `score: f32` parameter — real
//!   Lucene's own `Collector`/`LeafCollector` doesn't force `TotalHitCountCollector`
//!   to touch a `Scorer` either (`LeafCollector.setScorer` is a no-op there).
//!   Forcing a score parameter onto every collector would make the two
//!   existing, already-shipped, already-tested unscored collectors either grow
//!   a dummy parameter or get deleted for no correctness reason.
//! - **A trait per shape, not one trait doing double duty.** `Collector` and
//!   `ScoringCollector` are different contracts (`fn(i32)` vs `fn(i32, f32)`);
//!   giving them different trait names (as opposed to one trait with both
//!   methods, one of them defaulted to a no-op) keeps each collector's impl
//!   exactly as small as the contract it actually fulfills, and keeps
//!   [`search_term_query`]/[`search_boolean_query`]'s existing generic bound
//!   (`C: Collector`) untouched — no existing caller's code breaks.
//! - **The cost is one more trait, not a hierarchy.** With exactly two shapes
//!   (unscored / scored) and no third on the horizon, this is the same
//!   "don't build the trait hierarchy until a second real shape needs it"
//!   call `lib.rs`'s module doc already made for `Weight`/`Scorer` — here the
//!   second shape *has* arrived, so it gets its own trait, but no further
//!   speculative generality (no shared supertrait, no `Collector: ScoringCollector`
//!   blanket impl) is introduced beyond that.

/// Called once per matching, live doc ID, in ascending order — the entire
/// contract a collector needs for this slice (no scores, no per-segment
/// rebinding, no early-termination signal yet; see module doc).
pub trait Collector {
    fn collect(&mut self, doc_id: i32);
}

/// Called once per matching, live doc ID, in ascending-by-doc-ID order, with
/// that document's relevance score attached — the scored sibling of
/// [`Collector`] (see this module's doc comment for why it's a separate trait
/// rather than a breaking change to `Collector`).
pub trait ScoringCollector {
    fn collect(&mut self, doc_id: i32, score: f32);
}

/// Collects every matching doc ID into a `Vec<i32>`, ascending — the
/// `TopDocs`-shaped "give me the actual hits" collector.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VecCollector {
    pub docs: Vec<i32>,
}

impl Collector for VecCollector {
    fn collect(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }
}

/// `TotalHitCountCollector`-equivalent: counts matches without retaining doc
/// IDs, for callers that only need "how many docs match".
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CountCollector {
    pub count: i32,
}

impl Collector for CountCollector {
    fn collect(&mut self, _doc_id: i32) {
        self.count += 1;
    }
}

/// One scored hit: `ScoreDoc`-equivalent (`org.apache.lucene.search.ScoreDoc`),
/// minus the `shardIndex` field (meaningless — this port has no multi-shard
/// federation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreDoc {
    pub doc_id: i32,
    pub score: f32,
}

/// Real Lucene's ranking order: higher score first, and — verified against
/// `HitQueue.lessThan` (`org.apache.lucene.search.HitQueue`, not assumed —
/// `hitA.score == hitB.score ? hitA.doc > hitB.doc : hitA.score < hitB.score`,
/// i.e. on an exact score tie the *lower* doc ID is considered the better hit)
/// — **lower doc ID wins a score tie**. Returns `Ordering::Greater` when `a`
/// should rank ahead of `b`.
fn rank_order(a: &ScoreDoc, b: &ScoreDoc) -> std::cmp::Ordering {
    match a.score.total_cmp(&b.score) {
        std::cmp::Ordering::Equal => b.doc_id.cmp(&a.doc_id),
        other => other,
    }
}

/// `TopScoreDocCollector`-equivalent: keeps the top `n` `(doc_id, score)` hits
/// by score (ties broken by lower doc ID, matching real Lucene's `HitQueue` —
/// see [`rank_order`]), discarding everything else.
///
/// **Design**: real `TopScoreDocCollector` is backed by a `HitQueue` (a binary
/// min-heap over the *worst* currently-kept hit, so a new hit only needs one
/// comparison against the heap's root to know whether it's worth keeping).
/// This port instead keeps `hits` fully sorted (best-first) after every
/// insert/eviction — a plain `Vec` with a binary-search insert position. This
/// is the same tradeoff this crate's `docid_set` module already made for
/// `Disjunction` ("simple first cut, revisit if scale demands it" — see that
/// module's doc comment): correct, `O(n)` per insert instead of `O(log n)`,
/// fine for the query sizes and `top_n` values this port's fixtures and tests
/// exercise today.
#[derive(Debug, Clone)]
pub struct TopDocsCollector {
    top_n: usize,
    hits: Vec<ScoreDoc>,
}

impl TopDocsCollector {
    /// A collector that keeps at most `top_n` hits. `top_n == 0` is a defined
    /// "keep nothing" edge case (every `collect` call is a no-op), not a panic.
    pub fn new(top_n: usize) -> Self {
        Self {
            top_n,
            hits: Vec::new(),
        }
    }

    /// The kept hits, best-first (see [`rank_order`]) — `TopDocs.scoreDocs`-equivalent
    /// (this port has no separate `totalHits`/`TotalHits.Relation` tracking, since
    /// nothing here does early termination yet; every `collect` call is a real
    /// evaluated hit).
    pub fn top_docs(&self) -> &[ScoreDoc] {
        &self.hits
    }
}

impl ScoringCollector for TopDocsCollector {
    fn collect(&mut self, doc_id: i32, score: f32) {
        if self.top_n == 0 {
            return;
        }
        let candidate = ScoreDoc { doc_id, score };
        if self.hits.len() < self.top_n {
            let pos = self
                .hits
                .partition_point(|h| rank_order(h, &candidate) == std::cmp::Ordering::Greater);
            self.hits.insert(pos, candidate);
            return;
        }
        // Full: only replace the current worst (last) hit if the candidate outranks it.
        if let Some(worst) = self.hits.last() {
            if rank_order(&candidate, worst) == std::cmp::Ordering::Greater {
                self.hits.pop();
                let pos = self
                    .hits
                    .partition_point(|h| rank_order(h, &candidate) == std::cmp::Ordering::Greater);
                self.hits.insert(pos, candidate);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_collector_collects_in_call_order() {
        let mut c = VecCollector::default();
        c.collect(3);
        c.collect(7);
        assert_eq!(c.docs, vec![3, 7]);
    }

    #[test]
    fn count_collector_counts_calls_not_values() {
        let mut c = CountCollector::default();
        c.collect(0);
        c.collect(0);
        c.collect(5);
        assert_eq!(c.count, 3);
    }

    fn score_docs(v: &[(i32, f32)]) -> Vec<ScoreDoc> {
        v.iter()
            .map(|&(doc_id, score)| ScoreDoc { doc_id, score })
            .collect()
    }

    #[test]
    fn top_docs_collector_empty_input_yields_no_hits() {
        let c = TopDocsCollector::new(3);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn top_docs_collector_top_n_zero_keeps_nothing() {
        let mut c = TopDocsCollector::new(0);
        c.collect(1, 5.0);
        c.collect(2, 9.0);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn top_docs_collector_fewer_than_n_keeps_all_sorted_by_score_desc() {
        let mut c = TopDocsCollector::new(5);
        c.collect(1, 1.0);
        c.collect(2, 3.0);
        c.collect(3, 2.0);
        assert_eq!(
            c.top_docs().to_vec(),
            score_docs(&[(2, 3.0), (3, 2.0), (1, 1.0)])
        );
    }

    #[test]
    fn top_docs_collector_exactly_n_keeps_all_sorted() {
        let mut c = TopDocsCollector::new(3);
        c.collect(1, 1.0);
        c.collect(2, 3.0);
        c.collect(3, 2.0);
        assert_eq!(
            c.top_docs().to_vec(),
            score_docs(&[(2, 3.0), (3, 2.0), (1, 1.0)])
        );
    }

    #[test]
    fn top_docs_collector_more_than_n_evicts_the_worst() {
        let mut c = TopDocsCollector::new(2);
        c.collect(1, 1.0);
        c.collect(2, 3.0);
        c.collect(3, 2.0);
        // 1.0 (doc 1) is the worst score and gets evicted once a better candidate
        // (doc 3, score 2.0) arrives.
        assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 3.0), (3, 2.0)]));
    }

    #[test]
    fn top_docs_collector_candidate_worse_than_all_kept_hits_is_dropped() {
        let mut c = TopDocsCollector::new(2);
        c.collect(1, 5.0);
        c.collect(2, 4.0);
        c.collect(3, 1.0); // worse than both kept hits -- must not be kept.
        assert_eq!(c.top_docs().to_vec(), score_docs(&[(1, 5.0), (2, 4.0)]));
    }

    #[test]
    fn top_docs_collector_tie_break_prefers_lower_doc_id() {
        let mut c = TopDocsCollector::new(2);
        c.collect(5, 2.0);
        c.collect(2, 2.0);
        c.collect(9, 2.0);
        // All tied at score 2.0 -- lowest doc IDs (2, 5) must win over doc 9.
        assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 2.0), (5, 2.0)]));
    }

    #[test]
    fn top_docs_collector_tie_break_eviction_prefers_lower_doc_id() {
        let mut c = TopDocsCollector::new(1);
        c.collect(9, 3.0);
        c.collect(2, 3.0); // ties doc 9 on score; lower doc id must win.
        assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 3.0)]));
    }

    #[test]
    fn rank_order_orders_by_score_desc_then_doc_id_asc() {
        let a = ScoreDoc {
            doc_id: 1,
            score: 5.0,
        };
        let b = ScoreDoc {
            doc_id: 2,
            score: 5.0,
        };
        assert_eq!(rank_order(&a, &b), std::cmp::Ordering::Greater);
        assert_eq!(rank_order(&b, &a), std::cmp::Ordering::Less);
        let c = ScoreDoc {
            doc_id: 3,
            score: 6.0,
        };
        assert_eq!(rank_order(&c, &a), std::cmp::Ordering::Greater);
    }
}
