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

/// Ascending/descending toggle for [`TopFieldCollector`] — real Lucene's
/// `SortField.setReverse` flag, generalized to any numeric sort key (this
/// port's `SortField.Type.LONG`/`INT` support; see `doc_value_query`'s
/// `sort_top_n_by_numeric_doc_value` for how a `DOUBLE` field would map onto
/// this same `i64` key if a caller bit-reinterprets it, which this port
/// doesn't do yet — see `docs/parity.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// One ranked-by-field hit: a doc ID plus its already-decoded numeric sort
/// value — the `FieldDoc`-equivalent minimal shape (no `shardIndex`, same
/// simplification [`ScoreDoc`] already makes, and no secondary sort fields —
/// see [`TopFieldCollector`]'s doc comment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldValueDoc {
    pub doc_id: i32,
    pub value: i64,
}

/// Real Lucene's `FieldValueHitQueue`-equivalent ranking order: ranks by
/// `value` in `direction` (ascending or descending), and — on an exact value
/// tie — **lower doc ID wins**, the same tie-break convention
/// [`TopDocsCollector`]'s [`rank_order`] already documents for a BM25 score
/// tie, kept consistent here for a sort-value tie. Returns
/// `Ordering::Greater` when `a` should rank ahead of `b`.
fn field_rank_order(
    a: &FieldValueDoc,
    b: &FieldValueDoc,
    direction: SortDirection,
) -> std::cmp::Ordering {
    let value_order = match direction {
        // Ascending: the *smaller* value ranks ahead, so `a` ranking ahead of
        // `b` (Greater) happens when `a.value < b.value`, i.e. when
        // `b.value.cmp(&a.value)` is `Greater`.
        SortDirection::Ascending => b.value.cmp(&a.value),
        // Descending: the *larger* value ranks ahead -- direct `a.cmp(b)`.
        SortDirection::Descending => a.value.cmp(&b.value),
    };
    match value_order {
        std::cmp::Ordering::Equal => b.doc_id.cmp(&a.doc_id),
        other => other,
    }
}

/// `TopFieldCollector`-equivalent (`org.apache.lucene.search.TopFieldCollector`,
/// scoped to a single numeric `SortField`): keeps the top `n` `(doc_id, value)`
/// hits ranked by a numeric doc-value field, ascending or descending per
/// [`SortDirection`], ties broken by ascending doc ID (see [`field_rank_order`]),
/// discarding everything else.
///
/// **Scope**: numeric doc-value fields only (`SortField.Type.LONG`/`INT`,
/// via the `i64` key `value` already carries — a `DOUBLE` field's sort key
/// would need a bit-reinterpret step this port doesn't add yet). No String/
/// `SortedDocValues`-based sort, no multiple sort fields/secondary keys
/// beyond the single documented doc-ID tie-break. Missing-value handling
/// (a candidate doc with no value for the sort field) is the caller's job —
/// this collector only ever sees `(doc_id, value)` pairs a caller already
/// decided to `offer`; see `doc_value_query::MissingValue` for the policy
/// its composition functions apply before calling [`TopFieldCollector::offer`].
/// See `docs/parity.md` for the precise, honest scope statement.
///
/// **Design**: not a [`Collector`]/[`ScoringCollector`] impl, because neither
/// trait's `collect` signature can carry a `Result` for a doc-value decode
/// error, and reading a doc's sort value is a fallible operation (the same
/// reason `doc_value_query::sort_by_numeric_doc_value` is a standalone
/// function rather than a `Collector` variant, see that function's doc
/// comment). Composition functions (e.g.
/// `doc_value_query::sort_top_n_by_numeric_doc_value`) decode each candidate
/// doc's value themselves (propagating any decode error via `Result`) and
/// call [`TopFieldCollector::offer`] with the already-decoded `i64`, which is
/// infallible. Internally this is the exact same bounded, always-sorted
/// `Vec` design [`TopDocsCollector`] already uses (see that struct's doc
/// comment for the tradeoff rationale) — same `O(n)`-per-insert simple first
/// cut, revisit if scale demands it.
#[derive(Debug, Clone)]
pub struct TopFieldCollector {
    top_n: usize,
    direction: SortDirection,
    hits: Vec<FieldValueDoc>,
}

impl TopFieldCollector {
    /// A collector that keeps at most `top_n` hits ranked by `direction`.
    /// `top_n == 0` is a defined "keep nothing" edge case (every `offer` call
    /// is a no-op), not a panic.
    pub fn new(top_n: usize, direction: SortDirection) -> Self {
        Self {
            top_n,
            direction,
            hits: Vec::new(),
        }
    }

    /// Offers one already-decoded `(doc_id, value)` pair. Only inserted if it
    /// ranks ahead of the current worst kept hit (or there's still room) --
    /// see [`field_rank_order`].
    pub fn offer(&mut self, doc_id: i32, value: i64) {
        if self.top_n == 0 {
            return;
        }
        let candidate = FieldValueDoc { doc_id, value };
        if self.hits.len() < self.top_n {
            let pos = self.hits.partition_point(|h| {
                field_rank_order(h, &candidate, self.direction) == std::cmp::Ordering::Greater
            });
            self.hits.insert(pos, candidate);
            return;
        }
        if let Some(worst) = self.hits.last() {
            if field_rank_order(&candidate, worst, self.direction) == std::cmp::Ordering::Greater {
                self.hits.pop();
                let pos = self.hits.partition_point(|h| {
                    field_rank_order(h, &candidate, self.direction) == std::cmp::Ordering::Greater
                });
                self.hits.insert(pos, candidate);
            }
        }
    }

    /// The kept hits, best-first per [`SortDirection`] (see [`field_rank_order`]).
    pub fn top_docs(&self) -> &[FieldValueDoc] {
        &self.hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field_docs(v: &[(i32, i64)]) -> Vec<FieldValueDoc> {
        v.iter()
            .map(|&(doc_id, value)| FieldValueDoc { doc_id, value })
            .collect()
    }

    #[test]
    fn top_field_collector_empty_input_yields_no_hits() {
        let c = TopFieldCollector::new(3, SortDirection::Ascending);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn top_field_collector_top_n_zero_keeps_nothing() {
        let mut c = TopFieldCollector::new(0, SortDirection::Ascending);
        c.offer(1, 5);
        c.offer(2, 9);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn top_field_collector_ascending_orders_smallest_first() {
        let mut c = TopFieldCollector::new(5, SortDirection::Ascending);
        c.offer(1, 30);
        c.offer(2, 10);
        c.offer(3, 20);
        assert_eq!(
            c.top_docs().to_vec(),
            field_docs(&[(2, 10), (3, 20), (1, 30)])
        );
    }

    #[test]
    fn top_field_collector_descending_orders_largest_first() {
        let mut c = TopFieldCollector::new(5, SortDirection::Descending);
        c.offer(1, 30);
        c.offer(2, 10);
        c.offer(3, 20);
        assert_eq!(
            c.top_docs().to_vec(),
            field_docs(&[(1, 30), (3, 20), (2, 10)])
        );
    }

    #[test]
    fn top_field_collector_truncates_to_top_n_ascending() {
        let mut c = TopFieldCollector::new(2, SortDirection::Ascending);
        c.offer(1, 30);
        c.offer(2, 10);
        c.offer(3, 20);
        // Worst (doc 1, value 30) must be evicted, keeping the two smallest.
        assert_eq!(c.top_docs().to_vec(), field_docs(&[(2, 10), (3, 20)]));
    }

    #[test]
    fn top_field_collector_truncates_to_top_n_descending() {
        let mut c = TopFieldCollector::new(2, SortDirection::Descending);
        c.offer(1, 30);
        c.offer(2, 10);
        c.offer(3, 20);
        // Worst (doc 2, value 10) must be evicted, keeping the two largest.
        assert_eq!(c.top_docs().to_vec(), field_docs(&[(1, 30), (3, 20)]));
    }

    #[test]
    fn top_field_collector_tie_break_prefers_lower_doc_id() {
        let mut c = TopFieldCollector::new(2, SortDirection::Ascending);
        c.offer(5, 2);
        c.offer(2, 2);
        c.offer(9, 2);
        assert_eq!(c.top_docs().to_vec(), field_docs(&[(2, 2), (5, 2)]));
    }

    #[test]
    fn field_rank_order_ties_break_by_ascending_doc_id() {
        let a = FieldValueDoc {
            doc_id: 1,
            value: 5,
        };
        let b = FieldValueDoc {
            doc_id: 2,
            value: 5,
        };
        assert_eq!(
            field_rank_order(&a, &b, SortDirection::Ascending),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            field_rank_order(&a, &b, SortDirection::Descending),
            std::cmp::Ordering::Greater
        );
    }

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
