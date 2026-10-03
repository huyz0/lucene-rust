//! `DiversifyingChildrenFloatKnnVectorQuery` and
//! `DiversifyingChildrenByteKnnVectorQuery`: a KNN search over the children
//! of document blocks that keeps only the nearest child of each parent.
//!
//! The search itself is [`crate::vector_query`]'s -- the same per-leaf fan-out,
//! pro-rata optimistic collectors, re-entrant second pass and merge as
//! `KnnFloatVectorQuery` -- with each leaf's collector replaced by
//! `DiversifyingNearestChildrenKnnCollector` and its exact search by the
//! parent-grouping one. As with the plain KNN queries, the vector readers
//! and the child filter are inputs: each [`KnnSegment`]'s
//! [`VectorsInput::filter`](crate::vector_query::VectorsInput::filter) is the
//! child filter's matches in that segment
//! ([`filter_bitsets`](crate::vector_query::filter_bitsets)), `None` for no
//! filter. Hits are children, with global doc ids.

use std::sync::Arc;

use lucene_util::fixed_bit_set::FixedBitSet;

use super::BitSetProducer;
use crate::multi_segment::OpenSegment;
use crate::vector_query::{self, KnnByteVectorQuery, KnnFloatVectorQuery, KnnSegment};
use crate::{Error, Result, ScoreDoc};

/// `DiversifyingChildrenFloatKnnVectorQuery(field, query, childFilter, k,
/// parentsFilter)`: `query`'s field, target and `k`, over the children of
/// the blocks `parents` closes.
#[derive(Debug, Clone)]
pub struct DiversifyingChildrenFloatKnnVectorQuery {
    pub query: KnnFloatVectorQuery,
    pub parents: Arc<dyn BitSetProducer>,
}

/// `DiversifyingChildrenByteKnnVectorQuery`: the BYTE-encoded counterpart of
/// [`DiversifyingChildrenFloatKnnVectorQuery`].
#[derive(Debug, Clone)]
pub struct DiversifyingChildrenByteKnnVectorQuery {
    pub query: KnnByteVectorQuery,
    pub parents: Arc<dyn BitSetProducer>,
}

impl DiversifyingChildrenFloatKnnVectorQuery {
    pub fn new(query: KnnFloatVectorQuery, parents: Arc<dyn BitSetProducer>) -> Self {
        Self { query, parents }
    }

    /// `IndexSearcher.search(query, k)`'s hits: `segments[i]` and `knn[i]`
    /// are the same segment.
    pub fn search(
        &self,
        segments: &[OpenSegment<'_>],
        knn: &[KnnSegment<'_>],
    ) -> Result<Vec<ScoreDoc>> {
        let bits = parent_bit_sets(&*self.parents, segments, knn)?;
        let parents: Vec<Option<&FixedBitSet>> = bits.iter().map(|b| b.as_deref()).collect();
        vector_query::search_diversifying_children_float_knn_multi_segment(
            knn,
            &parents,
            &self.query,
        )
    }
}

impl DiversifyingChildrenByteKnnVectorQuery {
    pub fn new(query: KnnByteVectorQuery, parents: Arc<dyn BitSetProducer>) -> Self {
        Self { query, parents }
    }

    /// See [`DiversifyingChildrenFloatKnnVectorQuery::search`].
    pub fn search(
        &self,
        segments: &[OpenSegment<'_>],
        knn: &[KnnSegment<'_>],
    ) -> Result<Vec<ScoreDoc>> {
        let bits = parent_bit_sets(&*self.parents, segments, knn)?;
        let parents: Vec<Option<&FixedBitSet>> = bits.iter().map(|b| b.as_deref()).collect();
        vector_query::search_diversifying_children_byte_knn_multi_segment(
            knn,
            &parents,
            &self.query,
        )
    }
}

/// `parentsFilter.getBitSet(context)` for every leaf, after checking the two
/// views of the index line up.
fn parent_bit_sets(
    parents: &dyn BitSetProducer,
    segments: &[OpenSegment<'_>],
    knn: &[KnnSegment<'_>],
) -> Result<Vec<Option<Arc<FixedBitSet>>>> {
    if segments.len() != knn.len()
        || segments
            .iter()
            .zip(knn)
            .any(|(s, k)| s.doc_base != k.doc_base)
    {
        return Err(Error::IllegalArgument(
            "the open segments and the KNN segments must be the same leaves".into(),
        ));
    }
    segments.iter().map(|s| parents.bit_set(s)).collect()
}
