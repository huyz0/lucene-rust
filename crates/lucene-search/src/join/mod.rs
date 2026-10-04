//! `lucene-join`'s block joins (Lucene 10.5.0, `org.apache.lucene.search.join`):
//! queries over **document blocks** -- a run of child documents closed by
//! their parent, added together with `IndexWriter.addDocuments` and kept
//! contiguous by every flush and merge (`lucene_index`'s parent field keeps
//! them together under an index sort as well).
//!
//! - [`BitSetProducer`] / [`QueryBitSetProducer`]: the parent filter, one bit
//!   set per segment, cached per segment core.
//! - [`check_join_index`]: `CheckJoinIndex.check`, the structural check a
//!   block index must pass for the joins to be meaningful.
//!
//! A bit set here, as in Java, ignores deletions: a scorer never reads live
//! documents, so the parents of a segment are the filter's matches whether or
//! not they are deleted.

mod knn;
mod query;
mod sort;

pub use knn::{DiversifyingChildrenByteKnnVectorQuery, DiversifyingChildrenFloatKnnVectorQuery};

pub use query::{
    ParentChildrenBlockJoinQuery, ParentsChildrenBlockJoinQuery, ScoreCombiner, ScoreMode,
    ToChildBlockJoinQuery, ToParentBlockJoinQuery, DEFAULT_CHILD_LIMIT_PER_PARENT,
};
pub use sort::{BlockJoinSelector, JoinMissing, JoinSortType, ToParentBlockJoinSortField};

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::{Error, Result};

/// `BitSetProducer`: a per-segment bit set of documents (the parents of a
/// block join), `None` when the segment has none (Java's `null`).
pub trait BitSetProducer: fmt::Debug + Send + Sync {
    /// `getBitSet(context)`.
    fn bit_set(&self, leaf: &OpenSegment<'_>) -> Result<Option<Arc<FixedBitSet>>>;

    /// The identity `Query.equals`/`hashCode` give the producer (Java's
    /// `QueryBitSetProducer.equals` compares the wrapped query): two join
    /// queries over equal producers are equal, and the query cache tells
    /// them apart by it.
    fn key(&self) -> String;
}

/// A producer's cache: a segment core (its name and id) to its bit set.
type BitSetCache = HashMap<(String, [u8; 16]), Option<Arc<FixedBitSet>>>;

/// `QueryBitSetProducer`: the documents `query` matches, as a bit set per
/// segment, computed once per segment core and cached (Java keys a
/// `WeakHashMap` on `getCoreCacheHelper().getKey()`; here the segment's name
/// and id, which identify its core -- the cache holds one entry per segment
/// it has seen, until [`QueryBitSetProducer::clear`]).
///
/// The query runs as Java runs it: rewritten, `COMPLETE_NO_SCORES`, without
/// a query cache, and its scorer's iterator taken whole -- deleted documents
/// included.
pub struct QueryBitSetProducer {
    query: BooleanQuery,
    cache: Mutex<BitSetCache>,
}

impl QueryBitSetProducer {
    /// `new QueryBitSetProducer(query)`.
    pub fn new(query: BooleanQuery) -> Self {
        Self {
            query,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// `getQuery()`.
    pub fn query(&self) -> &BooleanQuery {
        &self.query
    }

    /// Drops every cached segment (Java's entries go when their segment
    /// core is collected). Nothing here evicts on its own: a producer that
    /// outlives reader reopens keeps the bit set of every segment it ever
    /// saw, merged-away ones included, until this is called -- a long-lived
    /// producer should be cleared (or rebuilt) when its reader is replaced.
    pub fn clear(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BitSetCache> {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl fmt::Debug for QueryBitSetProducer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QueryBitSetProducer({:?})", self.query)
    }
}

impl BitSetProducer for QueryBitSetProducer {
    fn bit_set(&self, leaf: &OpenSegment<'_>) -> Result<Option<Arc<FixedBitSet>>> {
        let key = leaf
            .reader
            .map(|r| (r.segment_name.clone(), r.segment_id()));
        if let Some(key) = &key {
            if let Some(hit) = self.lock().get(key) {
                return Ok(hit.clone());
            }
        }
        let bits = query_bit_set(leaf, &self.query)?.map(Arc::new);
        if let Some(key) = key {
            self.lock().insert(key, bits.clone());
        }
        Ok(bits)
    }

    fn key(&self) -> String {
        format!("{self:?}")
    }
}

/// `BitSet.of(weight.scorer(context).iterator(), maxDoc)` for `query` over
/// one segment, deletions not applied; `None` without a scorer.
fn query_bit_set(leaf: &OpenSegment<'_>, query: &BooleanQuery) -> Result<Option<FixedBitSet>> {
    let max_doc = leaf
        .max_doc
        .or(leaf.reader.map(|r| r.max_doc))
        .ok_or_else(|| Error::IllegalArgument("a bit set needs the segment's maxDoc".into()))?;
    let len = usize::try_from(max_doc).unwrap_or(0);
    let one = OpenSegment {
        live_docs: None,
        cache: None,
        max_doc: Some(max_doc),
        ..*leaf
    };
    let rewritten = crate::multi_segment::rewrite_points_ranges(query, std::slice::from_ref(&one));
    let query = rewritten.as_ref().unwrap_or(query);
    let clause = crate::aggs::lone_clause(query);
    let ctx = crate::aggs::plain_context(&one);
    let mut buf = Vec::new();
    let docs = match crate::aggs::segment_matches(&ctx, query, &clause, None, &mut buf)? {
        None => return Ok(None),
        Some(Some(docs)) => docs,
        Some(None) => {
            let mut all = FixedBitSet::new(len);
            all.set_range(0, len);
            return Ok(Some(all));
        }
    };
    let mut bits = FixedBitSet::new(len);
    for &doc in docs {
        if let Ok(i) = usize::try_from(doc) {
            // FBS: a scorer returns this segment's documents, below
            // `maxDoc`, the set's length; the check keeps a corrupt one from
            // panicking.
            if i < bits.len() {
                bits.set(i);
            }
        }
    }
    Ok(Some(bits))
}

/// `CheckJoinIndex.check(reader, parentsFilter)`: every non-empty segment has
/// at least one parent, ends in a parent, and -- when it has deletions --
/// deletes each block whole (a parent and its children are live or deleted
/// together). The first violation is an [`Error::IllegalState`] with Java's
/// message.
pub fn check_join_index(segments: &[OpenSegment<'_>], parents: &dyn BitSetProducer) -> Result<()> {
    for (ord, seg) in segments.iter().enumerate() {
        let max_doc = seg.max_doc.or(seg.reader.map(|r| r.max_doc)).unwrap_or(0);
        if max_doc == 0 {
            continue;
        }
        let name = seg
            .reader
            .map_or_else(|| format!("segment {ord}"), |r| r.segment_name.clone());
        let bits = parents.bit_set(seg)?;
        let Some(bits) = bits.filter(|b| b.cardinality() > 0) else {
            return Err(Error::IllegalState(format!(
                "Every segment should have at least one parent, but {name} does not have any"
            )));
        };
        let last = usize::try_from(max_doc.saturating_sub(1)).unwrap_or(0);
        // FBS: `last < maxDoc`; a producer's set shorter than the segment is
        // read as "not a parent".
        if last >= bits.len() || !bits.get(last) {
            return Err(Error::IllegalState(format!(
                "The last document of a segment must always be a parent, but {name} has a \
                 child as a last doc"
            )));
        }
        let Some(live) = seg.live_docs else {
            continue;
        };
        let mut prev_parent: usize = 0;
        let mut first = true;
        let mut parent = bits.next_set_bit(0);
        while let Some(p) = parent {
            // FBS: `p` is a bit of `bits`; `live` covers the segment.
            let parent_live = p < live.len() && live.get(p);
            let start = if first {
                0
            } else {
                prev_parent.saturating_add(1)
            };
            for child in start..p {
                // FBS: `child < p < bits.len()`, checked against `live`.
                let child_live = child < live.len() && live.get(child);
                if parent_live != child_live {
                    return Err(Error::IllegalState(if parent_live {
                        format!(
                            "Parent doc {p} of segment {name} is live but has a deleted child \
                             document {child}"
                        )
                    } else {
                        format!(
                            "Parent doc {p} of segment {name} is deleted but has a live child \
                             document {child}"
                        )
                    }));
                }
            }
            first = false;
            prev_parent = p;
            parent = p.checked_add(1).and_then(|n| bits.next_set_bit(n));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
