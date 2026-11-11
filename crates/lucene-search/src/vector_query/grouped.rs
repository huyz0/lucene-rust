//! KNN grouped by a field: the best `per_group` documents of each of the
//! best `k` groups a per-document group id sorts a segment's documents into
//! (Milvus' `group_by_field` with `group_size`, Qdrant's `groups`). The
//! graph walk collects into a [`GroupTop`] as the block-join search collects
//! into `DiversifyingNearestChildrenKnnCollector`, and a filter too
//! selective for the graph takes an exhaustive pass, as there.

use lucene_util::group_top::{GroupItem, GroupTop};

use super::{
    java_float_compare, plan_leaves, search_leaf, AcceptOrds, Diversify, KnnByteVectorQuery,
    KnnFloatVectorQuery, KnnQuery, KnnSegment, LeafExtras, LeafGraph, LeafHits, LeafPlan,
    SearchOptions,
};
use crate::collector::ScoreDoc;
use crate::Result;
use lucene_codecs::hnsw::{KnnCollect, VectorScorer};
use lucene_util::fixed_bit_set::FixedBitSet;

/// How one segment's documents fall into groups.
#[derive(Clone, Copy)]
pub struct SegmentGroups<'a> {
    /// A document's group.
    pub group: &'a dyn Fn(i32) -> i32,
    /// How many groups the segment's documents fall into, or more: it sizes
    /// the candidate lists and tells when every group was found.
    pub groups: usize,
    /// The documents kept of each group, at least one.
    pub per_group: usize,
}

impl std::fmt::Debug for SegmentGroups<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentGroups")
            .field("groups", &self.groups)
            .field("per_group", &self.per_group)
            .finish_non_exhaustive()
    }
}

/// A collected document: ordered by score (`Float.compare`), then the lower
/// document first, as the block-join collector orders its children.
#[derive(Debug, Clone, Copy)]
pub(super) struct GroupHit {
    pub(super) score: f32,
    pub(super) doc: i32,
    pub(super) ord: i32,
}

impl PartialEq for GroupHit {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o).is_eq()
    }
}
impl Eq for GroupHit {}
impl PartialOrd for GroupHit {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for GroupHit {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        java_float_compare(self.score, o.score).then_with(|| o.doc.cmp(&self.doc))
    }
}
impl GroupItem for GroupHit {
    fn doc(&self) -> i32 {
        self.doc
    }
}

/// The graph walk's collector: a [`GroupTop`] of `k` groups behind the
/// reader's ordinal-to-document translation. Its competitive bound is the
/// worst document the `k` groups hold, once `k` are held.
struct GroupedCollector<'f> {
    /// The walk's beam: the groups wanted times the documents of each.
    beam: usize,
    visit_limit: u64,
    visited: u64,
    ord_to_doc: &'f dyn Fn(i32) -> Result<i32>,
    groups: SegmentGroups<'f>,
    top: GroupTop<GroupHit>,
    error: Option<crate::Error>,
}

impl<'f> GroupedCollector<'f> {
    fn new(
        k: usize,
        visit_limit: u64,
        groups: SegmentGroups<'f>,
        ord_to_doc: &'f dyn Fn(i32) -> Result<i32>,
    ) -> Self {
        Self {
            beam: k.saturating_mul(groups.per_group.max(1)),
            visit_limit,
            visited: 0,
            ord_to_doc,
            groups,
            top: GroupTop::new(k, groups.per_group),
            error: None,
        }
    }
}

impl KnnCollect for GroupedCollector<'_> {
    fn k(&self) -> usize {
        self.beam
    }
    fn early_terminated(&self) -> bool {
        self.visited >= self.visit_limit
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.visited = self.visited.saturating_add(count as u64);
    }
    fn visited_count(&self) -> u64 {
        self.visited
    }
    fn visit_limit(&self) -> u64 {
        self.visit_limit
    }
    fn collect(&mut self, ord: i32, similarity: f32) -> bool {
        let doc = match (self.ord_to_doc)(ord) {
            Ok(d) => d,
            Err(e) => {
                self.error.get_or_insert(e);
                return false;
            }
        };
        let group = (self.groups.group)(doc);
        self.top.offer(
            group,
            GroupHit {
                score: similarity,
                doc,
                ord,
            },
        )
    }
    fn min_competitive_similarity(&self) -> f32 {
        self.top.floor().map_or(f32::NEG_INFINITY, |h| h.score)
    }
}

/// Every accepted ordinal scored, each group's best `per_group` kept, the
/// best `k` groups returned: the exact answer, whatever the order.
fn grouped_exact<S: VectorScorer>(
    scorer: &mut S,
    accept_ords: Option<&FixedBitSet>,
    k: usize,
    groups: SegmentGroups<'_>,
    ord_to_doc: &dyn Fn(i32) -> Result<i32>,
) -> Result<Vec<GroupHit>> {
    // Every group held while scoring, so none is dropped before its best
    // document is seen; the best `k` are picked after.
    let mut top = GroupTop::new(usize::MAX, groups.per_group);
    let end = scorer.max_ord().max(0);
    for ord in 0..end {
        if accept_ords.is_some_and(|b| !b.get_doc(ord)) {
            continue;
        }
        let doc = ord_to_doc(ord)?;
        let score = scorer.score(ord)?;
        top.offer((groups.group)(doc), GroupHit { score, doc, ord });
    }
    Ok(best_groups(top, k))
}

/// The documents of `top`'s best `k` groups, best first.
fn best_groups(top: GroupTop<GroupHit>, k: usize) -> Vec<GroupHit> {
    let mut kept = 0usize;
    let mut last = None;
    let mut out: Vec<GroupHit> = top
        .into_groups()
        .into_iter()
        .take_while(|&(g, _)| {
            if last != Some(g) {
                last = Some(g);
                kept = kept.saturating_add(1);
            }
            kept <= k
        })
        .map(|(_, h)| h)
        .collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

/// A walk of the graph into a [`GroupedCollector`]: its documents, best
/// first, the groups it held, and whether it stopped at the visit limit.
#[allow(clippy::too_many_arguments)]
fn grouped_approximate<S: VectorScorer>(
    scorer: &mut S,
    graph: &LeafGraph<'_, '_>,
    k: usize,
    limit: u64,
    options: SearchOptions<'_>,
    groups: SegmentGroups<'_>,
    ord_to_doc: &dyn Fn(i32) -> Result<i32>,
) -> Result<(Vec<GroupHit>, usize, bool)> {
    // `Lucene90HnswVectorsReader.search` collects documents already.
    let identity = |o: i32| -> Result<i32> { Ok(o) };
    let translate: &dyn Fn(i32) -> Result<i32> = if graph.hits_are_ordinals() {
        &identity
    } else {
        ord_to_doc
    };
    let mut c = GroupedCollector::new(k, limit, groups, translate);
    if scorer.max_ord() > 0 && k > 0 {
        graph.search_with(scorer, &mut c, options)?;
    }
    if let Some(e) = c.error.take() {
        return Err(e);
    }
    let early = KnnCollect::early_terminated(&c);
    let held = c.top.len();
    Ok((c.top.into_sorted(), held, early))
}

/// One leaf's grouped hits, local documents best first: the graph walked
/// (a filtered leaf as `getLeafResults` walks it, its visits bounded by the
/// filter's count), or every accepted vector scored when the filter passes
/// no more vectors than the groups want, or the walk stopped short of them.
#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_leaf_results<S: VectorScorer>(
    scorer: &mut S,
    graph: &LeafGraph<'_, '_>,
    accept: Option<&AcceptOrds<'_>>,
    ord_to_doc: &dyn Fn(i32) -> Result<i32>,
    max_doc: i32,
    plan: &LeafPlan,
    groups: SegmentGroups<'_>,
) -> Result<(LeafHits, bool)> {
    let size = scorer.max_ord().max(0) as usize;
    let k = plan.collector_k.min(size);
    let wanted = plan.k.min(size).min(groups.groups.max(1));
    let accept_bits = accept.map(|a| a.bits());
    let (hits, early) = if !plan.filtered {
        let (hits, _, early) = grouped_approximate(
            scorer,
            graph,
            k,
            plan.visited_limit,
            SearchOptions {
                accept_ords: accept_bits,
                filtered_doc_count: Some(max_doc),
                seed_ords: None,
                filtered_search_threshold: plan.extras.filtered_search_threshold,
            },
            groups,
            ord_to_doc,
        )?;
        (hits, early)
    } else {
        let bits = accept_bits.expect("a filtered leaf always has an accept set");
        let cost = bits.cardinality();
        if cost <= plan.k.saturating_mul(groups.per_group.max(1)) {
            (
                grouped_exact(scorer, Some(bits), k, groups, ord_to_doc)?,
                false,
            )
        } else {
            let limit = plan.visited_limit.min(cost as u64 + 1);
            let (hits, held, early) = grouped_approximate(
                scorer,
                graph,
                k,
                limit,
                SearchOptions {
                    accept_ords: Some(bits),
                    filtered_doc_count: Some(cost as i32),
                    seed_ords: None,
                    filtered_search_threshold: plan.extras.filtered_search_threshold,
                },
                groups,
                ord_to_doc,
            )?;
            if (!early && held >= wanted) || plan.extras.timed_out() {
                (hits, early)
            } else {
                (
                    grouped_exact(scorer, Some(bits), k, groups, ord_to_doc)?,
                    false,
                )
            }
        }
    };
    let mut ords: Vec<i32> = hits.iter().map(|h| h.ord).collect();
    ords.sort_unstable();
    Ok((
        LeafHits {
            hits: hits
                .into_iter()
                .map(|h| ScoreDoc {
                    doc_id: h.doc,
                    score: h.score,
                })
                .collect(),
            ords,
        },
        early,
    ))
}

/// The best `per_group` documents of each of the best `query.k` groups of
/// every segment, each segment on its own: `groups` has one entry per
/// segment, `None` for a segment none of whose documents is grouped (no
/// results there). Hits are local documents, best first, per segment; the
/// caller merges them by the groups' values, which only it knows.
///
/// # Errors
/// As [`super::search_knn_float_vector_query_multi_segment`].
pub fn search_grouped_float_knn_per_segment(
    segments: &[KnnSegment<'_>],
    groups: &[Option<SegmentGroups<'_>>],
    query: &KnnFloatVectorQuery,
) -> Result<Vec<Vec<ScoreDoc>>> {
    grouped_per_segment(segments, groups, query)
}

/// [`search_grouped_float_knn_per_segment`] for a `BYTE` field.
///
/// # Errors
/// As [`search_grouped_float_knn_per_segment`].
pub fn search_grouped_byte_knn_per_segment(
    segments: &[KnnSegment<'_>],
    groups: &[Option<SegmentGroups<'_>>],
    query: &KnnByteVectorQuery,
) -> Result<Vec<Vec<ScoreDoc>>> {
    grouped_per_segment(segments, groups, query)
}

fn grouped_per_segment<Q: KnnQuery>(
    segments: &[KnnSegment<'_>],
    groups: &[Option<SegmentGroups<'_>>],
    query: &Q,
) -> Result<Vec<Vec<ScoreDoc>>> {
    let (resolved, mut plans) = plan_leaves(segments, query, LeafExtras::default())?;
    let mut out = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let Some(g) = groups.get(i).copied().flatten() else {
            out.push(Vec::new());
            continue;
        };
        // Any segment may hold any group's best: each searches for every
        // group, not its share of them.
        let plan = &mut plans[i];
        plan.per_leaf_top_k = plan.k;
        plan.collector_k = plan.collector_k.max(plan.k);
        let (leaf, _) = search_leaf(
            &seg.vectors,
            &resolved[i],
            query.target(),
            plan,
            None,
            Some(Diversify::Groups(g)),
        )?;
        out.push(leaf.hits);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores ordinal `o` as `scores[o]`.
    struct Fixed(Vec<f32>);

    impl VectorScorer for Fixed {
        fn score(&mut self, ord: i32) -> lucene_codecs::vectors::Result<f32> {
            Ok(self.0[ord as usize])
        }
        fn max_ord(&self) -> i32 {
            self.0.len() as i32
        }
    }

    #[test]
    fn exact_keeps_each_groups_best_of_the_best_groups() {
        // Ordinal o is doc 10 + o, in group o % 3.
        let group = |d: i32| (d - 10) % 3;
        let doc_of = |o: i32| -> Result<i32> { Ok(o + 10) };
        let mut s = Fixed(vec![0.1, 0.9, 0.5, 0.8, 0.2, 0.7, 0.3]);
        let groups = SegmentGroups {
            group: &group,
            groups: 3,
            per_group: 2,
        };
        let hits = grouped_exact(&mut s, None, 2, groups, &doc_of).unwrap();
        // Group 1: ords 1 (0.9), 4 (0.2); group 2: 2 (0.5), 5 (0.7); group 0:
        // 3 (0.8), 0, 6 (0.3). Best groups by their best: 1 (0.9), 0 (0.8).
        let docs: Vec<i32> = hits.iter().map(|h| h.doc).collect();
        assert_eq!(docs, vec![11, 13, 16, 14]);
        let mut some = FixedBitSet::new(7);
        some.set(2);
        some.set(5);
        let hits = grouped_exact(&mut s, Some(&some), 2, groups, &doc_of).unwrap();
        assert_eq!(hits.iter().map(|h| h.doc).collect::<Vec<_>>(), vec![15, 12]);
        let broken = |_: i32| -> Result<i32> { Err(crate::Error::IllegalState("x".into())) };
        assert!(grouped_exact(&mut s, None, 2, groups, &broken).is_err());
    }

    #[test]
    fn collector_bounds_by_the_worst_held_document() {
        let group = |d: i32| d / 10;
        let doc_of = |o: i32| -> Result<i32> { Ok(o) };
        let groups = SegmentGroups {
            group: &group,
            groups: 10,
            per_group: 2,
        };
        let mut c = GroupedCollector::new(2, 3, groups, &doc_of);
        assert_eq!(KnnCollect::k(&c), 4);
        assert!(c.collect(1, 0.9));
        assert!(c.collect(2, 0.4));
        assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
        assert!(c.collect(11, 0.6));
        assert_eq!(c.min_competitive_similarity(), 0.4);
        // A third group must beat group 1's best.
        assert!(!c.collect(21, 0.5));
        assert!(c.collect(22, 0.7));
        assert_eq!(c.min_competitive_similarity(), 0.4);
        c.inc_visited_count(3);
        assert!(c.early_terminated() && c.visited_count() == 3 && c.visit_limit() == 3);
        let docs: Vec<i32> = c.top.into_sorted().iter().map(|h| h.doc).collect();
        assert_eq!(docs, vec![1, 22, 2]);
        let broken = |_: i32| -> Result<i32> { Err(crate::Error::IllegalState("x".into())) };
        let mut c = GroupedCollector::new(1, 3, groups, &broken);
        assert!(!c.collect(0, 1.0));
        assert!(c.error.is_some());
        let groups = SegmentGroups {
            group: &group,
            groups: 1,
            per_group: 1,
        };
        assert!(format!("{groups:?}").contains("per_group: 1"));
    }
}
