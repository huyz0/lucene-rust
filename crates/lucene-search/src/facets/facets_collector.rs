//! `FacetsCollector` and `FacetsCollectorManager`: the matching documents of
//! a search, per segment, for the facet implementations to count.
//!
//! A [`FacetsCollector`] is a [`SegmentCollector`]: the search hands it each
//! segment ([`FacetsCollector::set_next_reader`] /
//! [`SegmentCollector::finish`], for every segment, matching or not) and the
//! matching documents in order, which it keeps as a [`DocIdSet`] per segment
//! ([`MatchingDocs`]). [`FacetsCollectorManager::collect`] is
//! `searcher.search(query, facetsCollectorManager)`: one collector per slice,
//! then [`FacetsCollectorManager::reduce`].
//!
//! Differs:
//! * `reduce` merges the matching docs of a segment searched in several
//!   partitions into one, as Java does, but keeps the segments in the order
//!   they were first seen; Java collects them into a `HashMap` keyed by
//!   identity and returns its values in that map's (unspecified) order.
//! * `search`/`searchAfter` with a `Sort` are not ported (the field-sorted
//!   top docs collector is not driven here); the score-ordered ones are.

use std::borrow::Borrow;

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::TotalHitsRelation;
use crate::collector::{ScoreDoc, ScoreMode, ScoringCollector, TopDocsCollector, TotalHits};
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::{search_slices, PerSegment, SegmentCollector};
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::reader::{DocIdSetIterator, IndexReader, LeafReaderContext, NO_MORE_DOCS};
use crate::top_docs::{ShardScoreDoc, TopDocs};
use crate::{Error, Result};

/// A set of a segment's document ids: what `DocIdSetBuilder.build()` hands
/// out (`IntArrayDocIdSet` or `BitDocIdSet`), or `DocIdSet.EMPTY`.
#[derive(Debug, Clone)]
pub enum DocIdSet {
    /// `DocIdSet.EMPTY`.
    Empty,
    /// Sorted, distinct document ids.
    Array(Vec<i32>),
    /// A bit per document, and the number of documents set.
    Bits { bits: FixedBitSet, cost: i64 },
}

impl DocIdSet {
    /// `iterator()`.
    pub fn iterator(&self) -> DocIdSetIter<&DocIdSet> {
        DocIdSetIter {
            set: self,
            doc: -1,
            pos: 0,
        }
    }

    /// `iterator()` over a set the iterator owns.
    pub fn into_iterator(self) -> DocIdSetIter<DocIdSet> {
        DocIdSetIter {
            set: self,
            doc: -1,
            pos: 0,
        }
    }

    /// The number of documents in the set.
    pub fn len(&self) -> usize {
        match self {
            DocIdSet::Empty => 0,
            DocIdSet::Array(docs) => docs.len(),
            DocIdSet::Bits { bits, .. } => bits.cardinality(),
        }
    }

    /// Whether the set has no document.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// [`DocIdSet::iterator`]: over a borrowed or an owned set.
#[derive(Debug)]
pub struct DocIdSetIter<S: Borrow<DocIdSet>> {
    set: S,
    doc: i32,
    /// The next index into an [`DocIdSet::Array`].
    pos: usize,
}

impl<S: Borrow<DocIdSet>> DocIdSetIter<S> {
    fn exhaust(&mut self) -> i32 {
        self.doc = NO_MORE_DOCS;
        NO_MORE_DOCS
    }
}

impl<S: Borrow<DocIdSet>> DocIdSetIterator for DocIdSetIter<S> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc.saturating_add(1);
        self.advance(target)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.doc == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        let target = target.max(0);
        match self.set.borrow() {
            DocIdSet::Empty => Ok(self.exhaust()),
            DocIdSet::Array(docs) => {
                let mut pos = self.pos;
                while let Some(&d) = docs.get(pos) {
                    pos = pos.saturating_add(1);
                    if d >= target {
                        self.pos = pos;
                        self.doc = d;
                        return Ok(d);
                    }
                }
                self.pos = pos;
                Ok(self.exhaust())
            }
            DocIdSet::Bits { bits, .. } => {
                // `next_set_bit` is bounded by the bitset's own `len()`.
                let from = usize::try_from(target).unwrap_or(usize::MAX);
                let next = bits.next_set_bit(from).and_then(|d| i32::try_from(d).ok());
                match next {
                    Some(d) => {
                        self.doc = d;
                        Ok(d)
                    }
                    None => Ok(self.exhaust()),
                }
            }
        }
    }

    fn cost(&self) -> i64 {
        match self.set.borrow() {
            DocIdSet::Empty => 0,
            DocIdSet::Array(docs) => i64::try_from(docs.len()).unwrap_or(i64::MAX),
            DocIdSet::Bits { cost, .. } => *cost,
        }
    }
}

/// `DocIdSetBuilder(maxDoc)`: buffers ids, switching to a bitset once more
/// than `maxDoc >>> 7` were added.
#[derive(Debug)]
pub(crate) struct DocIdSetBuilder {
    max_doc: usize,
    threshold: usize,
    buffer: Vec<i32>,
    bits: Option<FixedBitSet>,
    /// Ids added since the switch to the bitset.
    counter: i64,
}

impl DocIdSetBuilder {
    pub(crate) fn new(max_doc: i32) -> Self {
        let max_doc = usize::try_from(max_doc).unwrap_or(0);
        DocIdSetBuilder {
            max_doc,
            threshold: max_doc >> 7,
            buffer: Vec::new(),
            bits: None,
            counter: 0,
        }
    }

    /// `grow(1).add(doc)`.
    #[inline]
    pub(crate) fn add(&mut self, doc: i32) {
        match &mut self.bits {
            Some(bits) => {
                // FBS: `doc` is a document of the segment this builder was
                // sized for; `get_doc`-style bounds keep a stray id out.
                if let Ok(i) = usize::try_from(doc) {
                    if i < bits.len() {
                        bits.set(i);
                    }
                }
                self.counter = self.counter.saturating_add(1);
            }
            None => {
                self.buffer.push(doc);
                if self.buffer.len() > self.threshold {
                    self.upgrade();
                }
            }
        }
    }

    /// `add(DocIdSetIterator)`.
    pub(crate) fn add_all(&mut self, it: &mut dyn DocIdSetIterator) -> Result<()> {
        loop {
            let d = it.next_doc()?;
            if d == NO_MORE_DOCS {
                return Ok(());
            }
            self.add(d);
        }
    }

    fn upgrade(&mut self) {
        let mut bits = FixedBitSet::new(self.max_doc);
        for &d in &self.buffer {
            if let Ok(i) = usize::try_from(d) {
                if i < bits.len() {
                    bits.set(i);
                }
            }
        }
        self.counter = i64::try_from(self.buffer.len()).unwrap_or(i64::MAX);
        self.buffer = Vec::new();
        self.bits = Some(bits);
    }

    /// `build()`.
    pub(crate) fn build(self) -> DocIdSet {
        match self.bits {
            Some(bits) => DocIdSet::Bits {
                bits,
                cost: self.counter,
            },
            None => {
                let mut docs = self.buffer;
                docs.sort_unstable();
                docs.dedup();
                DocIdSet::Array(docs)
            }
        }
    }
}

/// `FacetsCollector.MatchingDocs`: one segment's matching documents.
#[derive(Clone)]
pub struct MatchingDocs<'a> {
    /// `context`.
    pub context: LeafReaderContext<'a>,
    /// `bits`: the matching documents, segment-local.
    pub bits: DocIdSet,
    /// `totalHits`.
    pub total_hits: i32,
    /// `scores`, by segment-local document, when scores were kept.
    pub scores: Option<Vec<f32>>,
}

impl std::fmt::Debug for MatchingDocs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchingDocs")
            .field("ord", &self.context.ord)
            .field("doc_base", &self.context.doc_base)
            .field("total_hits", &self.total_hits)
            .field("bits", &self.bits)
            .finish()
    }
}

/// `FacetsCollector`.
#[derive(Default)]
pub struct FacetsCollector<'a> {
    keep_scores: bool,
    matching_docs: Vec<MatchingDocs<'a>>,
    context: Option<LeafReaderContext<'a>>,
    builder: Option<DocIdSetBuilder>,
    total_hits: i32,
    scores: Option<Vec<f32>>,
}

impl<'a> FacetsCollector<'a> {
    /// `new FacetsCollector(keepScores)`.
    pub fn new(keep_scores: bool) -> Self {
        FacetsCollector {
            keep_scores,
            ..Default::default()
        }
    }

    /// `getKeepScores()`.
    pub fn keep_scores(&self) -> bool {
        self.keep_scores
    }

    /// `getMatchingDocs()`.
    pub fn matching_docs(&self) -> &[MatchingDocs<'a>] {
        &self.matching_docs
    }

    /// `getMatchingDocs().add(..)`: what a caller collecting by hand appends.
    pub fn push_matching_docs(&mut self, docs: MatchingDocs<'a>) {
        self.matching_docs.push(docs);
    }
}

impl std::fmt::Debug for FacetsCollector<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetsCollector")
            .field("keep_scores", &self.keep_scores)
            .field("matching_docs", &self.matching_docs)
            .finish()
    }
}

impl<'a> SegmentCollector<'a> for FacetsCollector<'a> {
    fn score_mode(&self) -> ScoreMode {
        if self.keep_scores {
            ScoreMode::Complete
        } else {
            ScoreMode::CompleteNoScores
        }
    }

    /// `doSetNextReader(context)`.
    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        let reader = leaf
            .reader
            .ok_or_else(|| Error::MissingSegmentReader("FacetsCollector".into()))?;
        let context = LeafReaderContext {
            reader,
            ord,
            doc_base: leaf.doc_base,
        };
        self.builder = Some(DocIdSetBuilder::new(reader.max_doc()));
        self.total_hits = 0;
        if self.keep_scores {
            self.scores = Some(Vec::with_capacity(64));
        }
        self.context = Some(context);
        Ok(())
    }

    /// `collect(doc)`.
    #[inline]
    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        if let Some(b) = &mut self.builder {
            b.add(doc);
        }
        if let Some(scores) = &mut self.scores {
            if let Ok(i) = usize::try_from(doc) {
                if i >= scores.len() {
                    scores.resize(i.saturating_add(1), 0.0);
                }
                scores[i] = score;
            }
        }
        self.total_hits = self.total_hits.wrapping_add(1);
        Ok(())
    }

    /// `finish()`.
    fn finish(&mut self) -> Result<()> {
        let bits = match self.builder.take() {
            Some(b) => b.build(),
            None => DocIdSet::Empty,
        };
        let Some(context) = self.context.take() else {
            return Err(Error::IllegalState(
                "FacetsCollector.finish without a segment".into(),
            ));
        };
        self.matching_docs.push(MatchingDocs {
            context,
            bits,
            total_hits: self.total_hits,
            scores: self.scores.take(),
        });
        Ok(())
    }
}

/// `FacetsCollectorManager`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FacetsCollectorManager {
    keep_scores: bool,
}

/// `FacetsCollectorManager.FacetsResult`: the top documents and the facets
/// collector of one search.
#[derive(Debug)]
pub struct FacetsResult<'a> {
    pub top_docs: TopDocs,
    pub facets_collector: FacetsCollector<'a>,
}

impl FacetsCollectorManager {
    /// `new FacetsCollectorManager(keepScores)`.
    pub fn new(keep_scores: bool) -> Self {
        FacetsCollectorManager { keep_scores }
    }

    /// `newCollector()`.
    pub fn new_collector<'a>(&self) -> FacetsCollector<'a> {
        FacetsCollector::new(self.keep_scores)
    }

    /// `reduce(collectors)`: none is a new collector (that does not keep
    /// scores, as in Java), one is itself, several are merged segment by
    /// segment (see the module doc for the order).
    pub fn reduce<'a>(&self, collectors: Vec<FacetsCollector<'a>>) -> Result<FacetsCollector<'a>> {
        if collectors.len() <= 1 {
            return Ok(collectors
                .into_iter()
                .next()
                .unwrap_or_else(|| FacetsCollector::new(false)));
        }
        let mut merged: Vec<MatchingDocs<'a>> = Vec::new();
        for c in collectors {
            for docs in c.matching_docs {
                match merged
                    .iter_mut()
                    .find(|m| m.context.ord == docs.context.ord)
                {
                    Some(existing) => *existing = merge(existing, &docs)?,
                    None => merged.push(docs),
                }
            }
        }
        let mut reduced = FacetsCollector::new(self.keep_scores);
        reduced.matching_docs = merged;
        Ok(reduced)
    }

    /// `searcher.search(query, this)`: a collector per slice of `searcher`,
    /// reduced.
    pub fn collect<'a>(
        &self,
        searcher: &IndexSearcher<'_, 'a>,
        query: &BooleanQuery,
    ) -> Result<FacetsCollector<'a>> {
        let collectors = search_slices(searcher, query, || Ok(self.new_collector()))?;
        self.reduce(collectors)
    }

    /// `FacetsCollectorManager.search(searcher, q, n, fcm)`: the top `n`
    /// documents by score and the facets collector, in one pass.
    pub fn search<'a>(
        searcher: &IndexSearcher<'_, 'a>,
        query: &BooleanQuery,
        n: i32,
        fcm: &FacetsCollectorManager,
    ) -> Result<FacetsResult<'a>> {
        do_search(searcher, None, query, n, fcm)
    }

    /// `FacetsCollectorManager.searchAfter(searcher, after, q, n, fcm)`.
    pub fn search_after<'a>(
        searcher: &IndexSearcher<'_, 'a>,
        after: ScoreDoc,
        query: &BooleanQuery,
        n: i32,
        fcm: &FacetsCollectorManager,
    ) -> Result<FacetsResult<'a>> {
        do_search(searcher, Some(after), query, n, fcm)
    }
}

/// `FacetsCollectorManager.merge`: one segment's matching docs from two
/// partitions.
fn merge<'a>(a: &MatchingDocs<'a>, b: &MatchingDocs<'a>) -> Result<MatchingDocs<'a>> {
    let scores = match (&a.scores, &b.scores) {
        (None, None) => Some(Vec::new()),
        (Some(s), None) | (None, Some(s)) => Some(s.clone()),
        (Some(x), Some(y)) => {
            let len = x.len().max(y.len());
            Some(
                (0..len)
                    .map(|i| {
                        let first = x.get(i).copied().unwrap_or(0.0);
                        let second = y.get(i).copied().unwrap_or(0.0);
                        java_max(first, second)
                    })
                    .collect(),
            )
        }
    };
    let mut builder = DocIdSetBuilder::new(a.context.reader.max_doc());
    builder.add_all(&mut a.bits.iterator())?;
    builder.add_all(&mut b.bits.iterator())?;
    Ok(MatchingDocs {
        context: a.context,
        bits: builder.build(),
        total_hits: a.total_hits.wrapping_add(b.total_hits),
        scores,
    })
}

/// `Math.max(float, float)`: `NaN` wins, and `0.0` beats `-0.0`.
pub(crate) fn java_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a >= b {
        a
    } else {
        b
    }
}

/// The top documents collector and a slice's facets collector, fed together
/// (`MultiCollectorManager(topScoreDocs, fcm)`).
struct TopAndFacets<'c, 's, 'a> {
    top: &'c mut TopDocsCollector,
    facets: PerSegment<'c, 's, 'a, FacetsCollector<'a>>,
}

impl ScoringCollector for TopAndFacets<'_, '_, '_> {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.top.collect(doc_id, score);
        self.facets.collect(doc_id, score);
    }

    fn score_mode(&self) -> ScoreMode {
        ScoreMode::Complete
    }
}

/// `FacetsCollectorManager.doSearch` without a sort.
fn do_search<'a>(
    searcher: &IndexSearcher<'_, 'a>,
    after: Option<ScoreDoc>,
    query: &BooleanQuery,
    n: i32,
    fcm: &FacetsCollectorManager,
) -> Result<FacetsResult<'a>> {
    let mut limit = searcher.max_doc();
    if limit == 0 {
        limit = 1;
    }
    let n = n.min(limit);
    if let Some(a) = after {
        if a.doc_id >= limit {
            return Err(Error::IllegalArgument(format!(
                "after.doc exceeds the number of documents in the reader: after.doc={} limit={limit}",
                a.doc_id
            )));
        }
    }
    if n == 0 {
        let facets_collector = fcm.collect(searcher, query)?;
        let total = facets_collector
            .matching_docs()
            .iter()
            .fold(0i32, |t, m| t.wrapping_add(m.total_hits));
        return Ok(FacetsResult {
            top_docs: TopDocs {
                total_hits: TotalHits {
                    value: u64::try_from(total).unwrap_or(0),
                    relation: TotalHitsRelation::EqualTo,
                },
                score_docs: Vec::new(),
            },
            facets_collector,
        });
    }
    let Ok(top_n) = usize::try_from(n) else {
        return Err(Error::IllegalArgument(
            "numHits must be > 0; please use TotalHitCountCollectorManager if you just need the \
             total hit count"
                .into(),
        ));
    };
    // One top-docs collector across the slices: what merging each slice's
    // top docs (`TopDocs.merge`, ties by doc id) yields.
    let mut top = TopDocsCollector::with_total_hits_threshold(top_n, u64::MAX);
    if let Some(a) = after {
        top = top.with_after(a);
    }
    let mut collectors = Vec::new();
    let slices: Vec<Vec<usize>> = if searcher.slices().is_empty() {
        vec![Vec::new()]
    } else {
        searcher.slices().to_vec()
    };
    for slice in &slices {
        let mut fc = fcm.new_collector();
        {
            let mut both = TopAndFacets {
                top: &mut top,
                facets: PerSegment::over(searcher.segments(), slice, &mut fc),
            };
            searcher.search_slice_collector(query, slice, &mut both)?;
            both.facets.close()?;
        }
        collectors.push(fc);
    }
    Ok(FacetsResult {
        top_docs: TopDocs {
            total_hits: top.total_hits(),
            score_docs: top
                .top_docs()
                .iter()
                .map(|h| ShardScoreDoc::new(h.doc_id, h.score))
                .collect(),
        },
        facets_collector: fcm.reduce(collectors)?,
    })
}

#[cfg(test)]
mod tests;
