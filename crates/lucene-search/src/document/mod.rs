//! The queries, sort fields and value sources of Lucene's `document`
//! package: what `IntField.newRangeQuery`, `FeatureField.newSaturationQuery`,
//! `IntRange.newIntersectsQuery`, `SortedSetDocValuesField.newSlowRangeQuery`
//! and the rest of the package's factories build, run over the fields
//! `lucene_index::document` writes.
//!
//! # Shape
//!
//! Each Java query class is a struct implementing [`DocumentQuery`], whose
//! [`DocumentQuery::score_leaf`] is that query's `Weight.scorerSupplier(..)
//! .get(..)` driven by `DefaultBulkScorer` over one segment: every live match
//! in ascending doc-id order, with its score, into a [`ScoringCollector`].
//! [`DocumentQuery::rewrite`] is `Query.rewrite(IndexSearcher)`.
//! [`search_top_docs`] is `IndexSearcher.search(query, n)` over a reader's
//! segments.
//!
//! The factories keep Java's class/method names as module/function names:
//! `IntPoint.newRangeQuery(field, lo, hi)` is
//! [`int_point::new_range_query`]`(field, lo, hi)`, and so on. A factory that
//! wraps its query in a `BoostQuery` returns a [`Boosted`] query.
//!
//! # Deliberate differences
//!
//! - `IndexOrDocValuesQuery` picks the points side whenever the segment has
//!   the field's points (Java picks by cost; the hits and constant scores are
//!   the same either way) and the doc-values side otherwise.
//! - `IndexSortSortedNumericDocValuesRangeQuery`, which `IntField` and
//!   `LongField` wrap their range query in, is not applied: it only chooses a
//!   cheaper way to the same hits.
//! - `QueryVisitor`, `equals`/`hashCode`/`toString` and query caching are not
//!   ported.

mod distance_feature;
mod doc_values_queries;
mod feature;
mod point_queries;
mod range_queries;

use std::fmt;

use lucene_codecs::field_infos::FieldInfo;

use crate::collector::{LeafCollector, ScoreDoc, ScoringCollector, TopDocsCollector, TotalHits};
use crate::directory_reader::{ExistsDocs, SegmentReader};
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

pub use distance_feature::LongDistanceFeatureQuery;
pub use doc_values_queries::{
    keyword_field, numeric_doc_values_field, sorted_doc_values_field,
    sorted_numeric_doc_values_field, sorted_set_doc_values_field, DocValuesLongHashSet,
    DocValuesTermInSetQuery, RangeBulkScorer, SortedNumericDocValuesRangeQuery,
    SortedNumericDocValuesSetQuery, SortedSetDocValuesRangeQuery, SortedSetSelector,
    SortedSkipperScorerSupplier, TermConstantScoreQuery, TermInSetConstantScoreQuery,
};
pub use feature::{
    feature_field, FeatureDoubleValues, FeatureDoubleValuesSource, FeatureFunction, FeatureQuery,
    FeatureSortField,
};
pub use point_queries::{
    binary_point, double_field, double_point, float_field, float_point, inet_address_point,
    int_field, int_point, long_field, long_point, IndexOrDocValuesQuery, NumericSelector,
    PointInSetQuery, PointRangeQuery,
};
pub use range_queries::{
    double_range, double_range_doc_values_field, float_range, float_range_doc_values_field,
    inet_address_range, int_range, int_range_doc_values_field, long_range,
    long_range_doc_values_field, BinaryRangeDocValues, BinaryRangeFieldRangeQuery, RangeFieldQuery,
};

/// One document-package query: `Query.rewrite` plus the `Weight`'s scorer
/// over a segment. See the module doc.
pub trait DocumentQuery: fmt::Debug + Send + Sync {
    /// `Query.rewrite(IndexSearcher)` over the reader's `leaves`: `None` when
    /// the query rewrites to itself.
    fn rewrite(&self, leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        let _ = leaves;
        Ok(None)
    }

    /// Every live match of `leaf`, ascending by (leaf-local) doc id, with its
    /// score -- `boost` is the `boost` `createWeight` receives.
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()>;
}

/// `IndexSearcher.rewrite(query)`: rewritten until it stops changing.
pub fn rewrite(
    query: &dyn DocumentQuery,
    leaves: &[OpenSegment<'_>],
) -> Result<Option<Box<dyn DocumentQuery>>> {
    let mut current: Option<Box<dyn DocumentQuery>> = None;
    loop {
        let next = match &current {
            None => query.rewrite(leaves)?,
            Some(q) => q.rewrite(leaves)?,
        };
        match next {
            None => return Ok(current),
            Some(q) => current = Some(q),
        }
    }
}

/// `IndexSearcher.search(query, n)`'s answer.
#[derive(Debug, Clone, PartialEq)]
pub struct TopDocs {
    pub total_hits: TotalHits,
    /// Global doc ids, score descending, doc ascending on ties.
    pub score_docs: Vec<ScoreDoc>,
}

/// `IndexSearcher.search(query, n)` over `leaves` (in `doc_base` order):
/// rewritten, then each segment scored into one `TopScoreDocCollector`
/// (total hits exact up to 1,000, Java's default threshold).
pub fn search_top_docs(
    leaves: &[OpenSegment<'_>],
    query: &dyn DocumentQuery,
    n: usize,
) -> Result<TopDocs> {
    let rewritten = rewrite(query, leaves)?;
    let query: &dyn DocumentQuery = rewritten.as_deref().unwrap_or(query);
    let mut collector = TopDocsCollector::with_total_hits_threshold(n, 1000);
    for leaf in leaves {
        let mut lc = LeafCollector::new(&mut collector, leaf.doc_base);
        query.score_leaf(leaf, 1.0, &mut lc)?;
    }
    Ok(TopDocs {
        total_hits: collector.total_hits(),
        score_docs: collector.top_docs().to_vec(),
    })
}

/// Every match of `query` over `leaves`, as global doc ids with scores, in
/// doc-id order (`COMPLETE` score mode: nothing pruned).
pub fn search_all(leaves: &[OpenSegment<'_>], query: &dyn DocumentQuery) -> Result<Vec<ScoreDoc>> {
    let rewritten = rewrite(query, leaves)?;
    let query: &dyn DocumentQuery = rewritten.as_deref().unwrap_or(query);
    let mut all = AllHits::default();
    for leaf in leaves {
        let mut lc = LeafCollector::new(&mut all, leaf.doc_base);
        query.score_leaf(leaf, 1.0, &mut lc)?;
    }
    Ok(all.hits)
}

/// A collector keeping every hit.
#[derive(Debug, Default)]
struct AllHits {
    hits: Vec<ScoreDoc>,
}

impl ScoringCollector for AllHits {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.hits.push(ScoreDoc { doc_id, score });
    }
}

/// `BoostQuery`: `query` with its scores multiplied by `boost`, the way
/// `createWeight(searcher, scoreMode, boost)` passes it down.
#[derive(Debug)]
pub struct Boosted {
    pub query: Box<dyn DocumentQuery>,
    pub boost: f32,
}

impl Boosted {
    pub fn new(query: Box<dyn DocumentQuery>, boost: f32) -> Self {
        Boosted { query, boost }
    }
}

impl DocumentQuery for Boosted {
    /// `BoostQuery.rewrite`: the inner query rewritten, the boost kept.
    fn rewrite(&self, leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        Ok(self
            .query
            .rewrite(leaves)?
            .map(|q| Box::new(Boosted::new(q, self.boost)) as Box<dyn DocumentQuery>))
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        self.query.score_leaf(leaf, boost * self.boost, collector)
    }
}

/// `MatchNoDocsQuery`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchNoDocs;

impl DocumentQuery for MatchNoDocs {
    fn score_leaf(
        &self,
        _leaf: &OpenSegment<'_>,
        _boost: f32,
        _collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        Ok(())
    }
}

/// `MatchAllDocsQuery`: every live document, at the boost.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchAllDocs;

impl DocumentQuery for MatchAllDocs {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let max_doc = reader(leaf)?.max_doc;
        for doc in 0..max_doc {
            collect_live(leaf, doc, boost, collector);
        }
        Ok(())
    }
}

/// `FieldExistsQuery`: every live document with a value for `field` (its
/// norms, else its doc values), at the boost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldExists {
    pub field: String,
}

impl DocumentQuery for FieldExists {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let r = reader(leaf)?;
        match r.field_exists_docs(&self.field)? {
            ExistsDocs::None => {}
            ExistsDocs::All => {
                for doc in 0..r.max_doc {
                    collect_live(leaf, doc, boost, collector);
                }
            }
            ExistsDocs::Bits(bits) => {
                for doc in 0..r.max_doc {
                    if bits.get_doc(doc) {
                        collect_live(leaf, doc, boost, collector);
                    }
                }
            }
        }
        Ok(())
    }
}

/// The leaf's reader, which every document query needs.
pub(crate) fn reader<'a>(leaf: &OpenSegment<'a>) -> Result<&'a SegmentReader> {
    leaf.reader
        .ok_or_else(|| Error::DocumentQuery("a document query needs the segment's reader".into()))
}

/// The leaf's `FieldInfo` for `field`.
pub(crate) fn field_info<'a>(leaf: &OpenSegment<'a>, field: &str) -> Result<Option<&'a FieldInfo>> {
    Ok(reader(leaf)?
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == field))
}

/// Collects `doc` at `score` when it is live.
#[inline]
pub(crate) fn collect_live(
    leaf: &OpenSegment<'_>,
    doc: i32,
    score: f32,
    collector: &mut dyn ScoringCollector,
) {
    if leaf.live_docs.is_none_or(|bits| bits.get_doc(doc)) {
        collector.collect(doc, score);
    }
}

#[cfg(test)]
mod tests;
