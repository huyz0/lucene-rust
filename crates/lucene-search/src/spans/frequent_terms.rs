//! `SpanMultiTermQueryWrapper.FrequentTermsSpanBooleanQueryRewrite` (Lucene
//! 10.6, `branch_10_6`, #16484 with #16532's `PriorityQueue`): a span rewrite
//! that keeps only the `maxSize` most frequent terms a multi-term query
//! matches, ranked by reader-wide document frequency ([`FrequentTermsSpanBooleanQueryRewrite::DF_ORDER`]),
//! by that then total term frequency ([`FrequentTermsSpanBooleanQueryRewrite::DF_THEN_TTF_ORDER`]), or by a
//! caller's order.
//!
//! Java builds it on `ScoringRewrite`: every matching term of every segment
//! is collected first, its `TermStates` summed across the segments, then the
//! terms are offered in term order to a `PriorityQueue` of `maxSize` (no
//! max-clause check), which is popped least first into a `SpanOrQuery` of
//! `SpanTermQuery`s. This does the same, with Lucene's heap
//! ([`JavaPriorityQueue::insert_with_overflow_strict`]), so ties are kept and
//! popped in Java's order.
//!
//! Differs: the Rust [`SpanNode::MultiTerm`] has no `setRewriteMethod` (its
//! rewrite follows the wrapped query's, `selectRewriteMethod`), so this is
//! applied as Java's tests apply it, `rewrite(searcher, query)`, and its
//! [`SpanNode::Or`] placed in the span tree; the `SpanTermQuery`s do not
//! carry the collected `TermStates` (the weight looks the terms up again,
//! to the same statistics).

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::extended_query::MultiTermQuery;
use crate::index_searcher::IndexSearcher;
use crate::java_priority_queue::JavaPriorityQueue;
use crate::spans::SpanNode;
use crate::Result;

/// `FrequentTermsSpanBooleanQueryRewrite.ScoreTerm`: a collected term with
/// its reader-wide `TermStates` statistics, which an order ranks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreTerm {
    /// `term.field()`.
    pub field: String,
    /// `term.bytes()`.
    pub term: Vec<u8>,
    /// `termState.docFreq()`: summed over the segments.
    pub doc_freq: i32,
    /// `termState.totalTermFreq()`: summed over the segments.
    pub total_term_freq: i64,
}

/// A `Comparator<ScoreTerm>`: the least term is dropped first.
pub type ScoreTermOrder = fn(&ScoreTerm, &ScoreTerm) -> Ordering;

fn df_order(a: &ScoreTerm, b: &ScoreTerm) -> Ordering {
    a.doc_freq.cmp(&b.doc_freq)
}

fn df_then_ttf_order(a: &ScoreTerm, b: &ScoreTerm) -> Ordering {
    df_order(a, b).then_with(|| a.total_term_freq.cmp(&b.total_term_freq))
}

/// `SpanMultiTermQueryWrapper.FrequentTermsSpanBooleanQueryRewrite`.
#[derive(Debug, Clone, Copy)]
pub struct FrequentTermsSpanBooleanQueryRewrite {
    max_size: usize,
    order: ScoreTermOrder,
}

impl FrequentTermsSpanBooleanQueryRewrite {
    /// `DF_ORDER`: by document frequency.
    pub const DF_ORDER: ScoreTermOrder = df_order;
    /// `DF_THEN_TTF_ORDER`: by document frequency, ties by total term
    /// frequency.
    pub const DF_THEN_TTF_ORDER: ScoreTermOrder = df_then_ttf_order;

    /// `FrequentTermsSpanBooleanQueryRewrite(maxSize)`: ranked by
    /// [`Self::DF_ORDER`].
    pub fn new(max_size: usize) -> Self {
        Self::with_order(max_size, Self::DF_ORDER)
    }

    /// `FrequentTermsSpanBooleanQueryRewrite(maxSize, order)`.
    pub fn with_order(max_size: usize, order: ScoreTermOrder) -> Self {
        Self { max_size, order }
    }

    /// `getMaxSize()`.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// `rewrite(indexSearcher, query)`: the `maxSize` most frequent terms
    /// `query` matches in the searcher's reader, least frequent first, as a
    /// span disjunction of span terms.
    ///
    /// # Errors
    /// What reading the term dictionaries reports.
    pub fn rewrite(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        query: &MultiTermQuery,
    ) -> Result<SpanNode> {
        // `ScoringRewrite.rewrite`: `collectTerms` into a `BytesRefHash`
        // whose `TermStates` accumulate each segment's statistics, then the
        // terms in `BytesRef` order.
        let mut collected: BTreeMap<Vec<u8>, (i32, i64)> = BTreeMap::new();
        for seg in searcher.segments() {
            for (term, seeked) in
                crate::exec::extended::expand_terms(seg.fields, &query.source, None)?
            {
                let stats = collected.entry(term).or_insert((0, 0));
                stats.0 = stats.0.wrapping_add(seeked.stats.doc_freq);
                stats.1 = stats.1.wrapping_add(seeked.stats.total_term_freq);
            }
        }
        let order = self.order;
        // `getTopLevelBuilder()`: `lessThan(a, b)` is `order.compare(a, b) < 0`.
        let mut queue = JavaPriorityQueue::new(self.max_size, |a: &ScoreTerm, b: &ScoreTerm| {
            order(a, b) == Ordering::Less
        });
        let field = query.field();
        for (term, (doc_freq, total_term_freq)) in collected {
            // `addClause`: `insertWithOverflow(new ScoreTerm(term, states))`.
            queue.insert_with_overflow_strict(ScoreTerm {
                field: field.to_string(),
                term,
                doc_freq,
                total_term_freq,
            });
        }
        // `build`: popped least first.
        let mut clauses = Vec::with_capacity(queue.size());
        while let Some(st) = queue.pop() {
            clauses.push(SpanNode::term(st.field, st.term));
        }
        Ok(SpanNode::Or { clauses })
    }
}

impl PartialEq for FrequentTermsSpanBooleanQueryRewrite {
    /// `equals`: the same size and the same order (Java's `order.equals`,
    /// the comparator's identity).
    fn eq(&self, other: &Self) -> bool {
        self.max_size == other.max_size && std::ptr::fn_addr_eq(self.order, other.order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(term: &str, doc_freq: i32, total_term_freq: i64) -> ScoreTerm {
        ScoreTerm {
            field: "f".into(),
            term: term.as_bytes().to_vec(),
            doc_freq,
            total_term_freq,
        }
    }

    #[test]
    fn orders_and_equality() {
        let (a, b) = (st("a", 3, 9), st("b", 3, 4));
        let df = FrequentTermsSpanBooleanQueryRewrite::DF_ORDER;
        let ttf = FrequentTermsSpanBooleanQueryRewrite::DF_THEN_TTF_ORDER;
        assert_eq!(df(&a, &b), Ordering::Equal);
        assert_eq!(ttf(&a, &b), Ordering::Greater);
        assert_eq!(df(&st("c", 1, 1), &a), Ordering::Less);
        let r = FrequentTermsSpanBooleanQueryRewrite::new(8);
        assert_eq!(r.max_size(), 8);
        assert_eq!(r, FrequentTermsSpanBooleanQueryRewrite::with_order(8, df));
        assert_ne!(r, FrequentTermsSpanBooleanQueryRewrite::with_order(8, ttf));
        assert_ne!(r, FrequentTermsSpanBooleanQueryRewrite::new(7));
    }

    /// Lucene's `insertWithOverflow`: a tie with the least element is handed
    /// back, a greater one replaces it; an empty queue of size 0 keeps none.
    #[test]
    fn the_queue_keeps_ties_out_as_lucene_does() {
        let mut q = JavaPriorityQueue::new(2, |a: &i32, b: &i32| a < b);
        assert_eq!(q.insert_with_overflow_strict(5), None);
        assert_eq!(q.insert_with_overflow_strict(7), None);
        assert_eq!(q.insert_with_overflow_strict(5), Some(5));
        assert_eq!(q.insert_with_overflow_strict(4), Some(4));
        assert_eq!(q.insert_with_overflow_strict(6), Some(5));
        assert_eq!((q.pop(), q.pop(), q.pop()), (Some(6), Some(7), None));
        let mut none = JavaPriorityQueue::new(0, |a: &i32, b: &i32| a < b);
        assert_eq!(none.insert_with_overflow_strict(1), Some(1));
        assert_eq!(none.size(), 0);
    }
}
