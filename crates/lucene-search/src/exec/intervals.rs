//! `IntervalQuery`'s weight and `IntervalScorer` (`lucene-queries`,
//! `org.apache.lucene.queries.intervals`): the source's iterator over the
//! segment as a two-phase scorer -- the approximation is the iterator's
//! documents, a document matches when it has an interval -- scored by the
//! query's [`crate::intervals::IntervalScoreFunction`] over the sloppy
//! frequency of its intervals.

use super::{BoxScorer, LeafContext, Mode, Scorer, NO_MORE_DOCS};
use crate::explain::Explanation;
use crate::intervals::iterators::{self, BoxIntervals, IntervalIterator};
use crate::intervals::{IntervalQuery, IntervalScoreFunction, NO_MORE_INTERVALS};
use crate::Result;

/// `IntervalScorer`.
pub(crate) struct IntervalScorer<'a> {
    intervals: BoxIntervals<'a>,
    min_extent: i32,
    boost: f32,
    function: IntervalScoreFunction,
    freq: f32,
    last_scored_doc: i32,
}

impl<'a> IntervalScorer<'a> {
    fn new(
        intervals: BoxIntervals<'a>,
        min_extent: i32,
        boost: f32,
        function: IntervalScoreFunction,
    ) -> Self {
        IntervalScorer {
            intervals,
            min_extent,
            boost,
            function,
            freq: 0.0,
            last_scored_doc: -1,
        }
    }

    /// `ensureFreq()`: `freq += 1.0 / max(length - minExtent + 1, 1)` over
    /// the document's intervals, from the one `matches()` stopped on.
    fn ensure_freq(&mut self) -> Result<()> {
        let doc = self.intervals.doc_id();
        if self.last_scored_doc != doc {
            self.last_scored_doc = doc;
            self.freq = self.intervals.sum_freq(self.min_extent)?;
        }
        Ok(())
    }

    /// `freq()`.
    pub(crate) fn freq(&mut self) -> Result<f32> {
        self.ensure_freq()?;
        Ok(self.freq)
    }
}

impl Scorer for IntervalScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.intervals.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.intervals.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.intervals.advance(target)
    }
    fn cost(&self) -> i64 {
        self.intervals.cost()
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn matches(&mut self) -> Result<bool> {
        Ok(self.intervals.next_interval()? != NO_MORE_INTERVALS)
    }
    fn match_cost(&self) -> f32 {
        self.intervals.match_cost()
    }
    fn score(&mut self) -> Result<f32> {
        self.ensure_freq()?;
        Ok(self.function.score(self.boost, self.freq))
    }
    /// `getMaxScore`: the boost, the functions' bound.
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(self.boost)
    }
}

/// `IntervalWeight.scorerSupplier(context)`: `None` when the source has no
/// intervals for the field in this segment.
pub(crate) fn interval<'a>(
    ctx: &LeafContext<'a>,
    q: &IntervalQuery,
    boost: f32,
    _mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(it) = iterators::intervals(&q.source, &q.field, ctx)? else {
        return Ok(None);
    };
    Ok(Some(Box::new(IntervalScorer::new(
        it,
        q.source.min_extent(),
        boost,
        q.score_function,
    ))))
}

/// `IntervalWeight.explain(context, doc)`: the score function's
/// explanation of the document's sloppy frequency, or
/// `"no matching intervals"`. Deletions are not consulted, as Java's are
/// not.
pub(crate) fn explain_interval(
    ctx: &LeafContext<'_>,
    q: &IntervalQuery,
    boost: f32,
    doc: i32,
) -> Result<Explanation> {
    if let Some(it) = iterators::intervals(&q.source, &q.field, ctx)? {
        let mut scorer = IntervalScorer::new(it, q.source.min_extent(), boost, q.score_function);
        let mut d = scorer.advance(doc)?;
        while d != NO_MORE_DOCS && !scorer.matches()? {
            d = scorer.next_doc()?;
        }
        if d == doc {
            let freq = scorer.freq()?;
            return Ok(q.score_function.explain(&q.to_string(), boost, freq));
        }
    }
    Ok(Explanation::no_match("no matching intervals"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted iterator: per document, its intervals.
    struct Scripted {
        docs: Vec<(i32, Vec<(i32, i32)>)>,
        at: Option<usize>,
        upto: usize,
        current: (i32, i32),
    }

    impl iterators::IntervalIterator for Scripted {
        fn doc_id(&self) -> i32 {
            match self.at {
                None => -1,
                Some(i) => self.docs.get(i).map_or(NO_MORE_DOCS, |d| d.0),
            }
        }
        fn next_doc(&mut self) -> Result<i32> {
            let next = self.at.map_or(0, |i| i + 1);
            self.at = Some(next);
            self.upto = 0;
            self.current = (-1, -1);
            Ok(self.doc_id())
        }
        fn advance(&mut self, target: i32) -> Result<i32> {
            loop {
                let d = self.next_doc()?;
                if d >= target {
                    return Ok(d);
                }
            }
        }
        fn cost(&self) -> i64 {
            self.docs.len() as i64
        }
        fn start(&self) -> i32 {
            self.current.0
        }
        fn end(&self) -> i32 {
            self.current.1
        }
        fn gaps(&self) -> i32 {
            0
        }
        fn next_interval(&mut self) -> Result<i32> {
            let ivs = &self.docs[self.at.unwrap_or(0)].1;
            self.current = ivs
                .get(self.upto)
                .copied()
                .unwrap_or((NO_MORE_INTERVALS, NO_MORE_INTERVALS));
            self.upto += 1;
            Ok(self.current.0)
        }
        fn match_cost(&self) -> f32 {
            3.5
        }
    }

    fn scripted(docs: Vec<(i32, Vec<(i32, i32)>)>) -> BoxIntervals<'static> {
        BoxIntervals::boxed(Scripted {
            docs,
            at: None,
            upto: 0,
            current: (-1, -1),
        })
    }

    /// The sloppy frequency sums `1 / max(width - minExtent + 1, 1)` in
    /// `double`, rounded to `float` each step; a document with no interval
    /// fails the second phase; scores go through the function.
    #[test]
    fn the_scorer_sums_sloppy_frequency_and_skips_documents_without_intervals() {
        let it = scripted(vec![
            (2, vec![(0, 0), (3, 5), (7, 8)]),
            (4, vec![]),
            (9, vec![(1, 1)]),
        ]);
        let f = IntervalScoreFunction::Saturation { pivot: 1.0 };
        let mut s = IntervalScorer::new(it, 2, 2.0, f);
        assert!(s.two_phase());
        assert_eq!(s.cost(), 3);
        assert_eq!(s.match_cost(), 3.5);
        assert_eq!(s.max_score(100).unwrap(), 2.0);
        assert_eq!(s.next_doc().unwrap(), 2);
        assert!(s.matches().unwrap());
        // widths 1, 3, 2 against minExtent 2: max(0,1)=1, 2, 1.
        let mut want = 0.0f32;
        for d in [1.0f64, 2.0, 1.0] {
            want = (f64::from(want) + 1.0 / d) as f32;
        }
        assert_eq!(s.freq().unwrap(), want);
        assert_eq!(s.score().unwrap(), 2.0 * (1.0 - 1.0 / (1.0 + want)));
        assert_eq!(s.next_doc().unwrap(), 4);
        assert!(!s.matches().unwrap());
        assert_eq!(s.advance(5).unwrap(), 9);
        assert!(s.matches().unwrap());
        assert_eq!(s.doc_id(), 9);
        assert_eq!(s.score().unwrap(), 2.0 * (1.0 - 1.0 / (1.0 + 1.0)));
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
    }
}
