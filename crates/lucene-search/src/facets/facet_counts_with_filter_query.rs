//! `FacetCountsWithFilterQuery`: the base of the facet counters that can
//! restrict their documents by a fast-match query (the range counters).

use super::facet_utils::ConjunctionIterator;
use super::facets_collector::{DocIdSet, MatchingDocs};
use crate::collector::ScoreMode;
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::{search_slice, SegmentCollector};
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::reader::DocIdSetIterator;
use crate::Result;

/// `FacetCountsWithFilterQuery`: holds the optional `fastMatchQuery`.
#[derive(Debug, Clone, Default)]
pub struct FacetCountsWithFilterQuery {
    fast_match_query: Option<BooleanQuery>,
}

impl FacetCountsWithFilterQuery {
    /// `new FacetCountsWithFilterQuery(fastMatchQuery)`.
    pub fn new(fast_match_query: Option<BooleanQuery>) -> Self {
        FacetCountsWithFilterQuery { fast_match_query }
    }

    /// `fastMatchQuery`.
    pub fn fast_match_query(&self) -> Option<&BooleanQuery> {
        self.fast_match_query.as_ref()
    }

    /// `createIterator(hits, iterators...)`: the matching documents of `hits`
    /// that every one of `iterators` has, and the fast-match query matches
    /// when there is one; `None` when any iterator is `None` or the query
    /// matches nothing in the segment (Java: has no scorer there).
    ///
    /// The query is run over the segment through `searcher` (Java builds a
    /// searcher over the segment's top-level reader), deleted documents
    /// skipped -- which `hits` already excludes.
    pub fn create_iterator<'i>(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        hits: &'i MatchingDocs<'_>,
        iterators: Vec<Option<Box<dyn DocIdSetIterator + 'i>>>,
    ) -> Result<Option<Box<dyn DocIdSetIterator + 'i>>> {
        let mut all: Vec<Box<dyn DocIdSetIterator + 'i>> = vec![Box::new(hits.bits.iterator())];
        for it in iterators {
            match it {
                Some(it) => all.push(it),
                None => return Ok(None),
            }
        }
        if let Some(query) = &self.fast_match_query {
            let mut docs = LeafDocs::default();
            search_slice(searcher, query, &[hits.context.ord], &mut docs)?;
            if docs.docs.is_empty() {
                return Ok(None);
            }
            all.push(Box::new(DocIdSet::Array(docs.docs).into_iterator()));
        }
        if all.len() == 1 {
            return Ok(all.pop());
        }
        Ok(Some(Box::new(ConjunctionIterator::new(all))))
    }
}

/// The documents a query matches in one segment.
#[derive(Default)]
struct LeafDocs {
    docs: Vec<i32>,
}

impl<'a> SegmentCollector<'a> for LeafDocs {
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }

    fn set_next_reader(&mut self, _ord: usize, _leaf: &OpenSegment<'a>) -> Result<()> {
        Ok(())
    }

    fn collect(&mut self, doc: i32, _score: f32) -> Result<()> {
        self.docs.push(doc);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
