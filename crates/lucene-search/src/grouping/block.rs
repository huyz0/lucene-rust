//! `BlockGroupingCollector`: groups that are blocks of adjacent documents
//! (indexed together with `addDocuments`), each closed by a document the
//! `lastDocPerGroup` query matches -- one pass, no group field.

use std::cmp::Ordering;

use lucene_util::fixed_bit_set::FixedBitSet;

use super::collectors::{GroupTopDocs, FLOAT_MIN_VALUE};
use super::sort::{compare_all, GroupSortValue, LeafKeys, Sort};
use super::{GroupDocs, GroupingCollectorManager, TopGroups};
use crate::collector::{ScoreMode, TotalHits, TotalHitsRelation};
use crate::join::query_time::java_max_f32;
use crate::leaf_collector::SegmentCollector;
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::{Error, Result};

/// `BlockGroupingCollector.OneGroup`: a group that made the queue.
struct OneGroup<'a> {
    leaf: OpenSegment<'a>,
    ord: usize,
    top_group_doc: i32,
    docs: Vec<i32>,
    scores: Vec<f32>,
    comparator_slot: usize,
}

/// `BlockGroupingCollector`: the top `topNGroups` blocks by `groupSort`
/// (each sorting as its best document), keeping every collected document
/// of each so [`Self::top_groups`] can sort them within the group.
///
/// The group-end documents are the `lastDocPerGroup` query's matches in
/// each segment, deleted documents included, as the weight's scorer gives
/// them.
pub struct BlockGroupingCollector<'a> {
    pending_sub_docs: Vec<i32>,
    pending_sub_scores: Vec<f32>,
    sort: Sort,
    reversed: Vec<i32>,
    top_n_groups: usize,
    last_doc_per_group: BooleanQuery,
    needs_scores: bool,
    /// The comparators' `topNGroups` slots.
    slots: Vec<Vec<GroupSortValue>>,
    bottom_slot: usize,
    queue_full: bool,
    current: Option<(usize, OpenSegment<'a>)>,
    top_group_doc: i32,
    total_hit_count: i32,
    total_group_count: i32,
    doc_base: i32,
    group_end_doc_id: i32,
    last_doc_per_group_bits: Option<FixedBitSet>,
    /// `groupQueue`, and which of its groups is its top (least competitive).
    queue: Vec<OneGroup<'a>>,
    queue_top: usize,
    group_competes: bool,
    keys: Option<LeafKeys<'a>>,
    scratch: Vec<GroupSortValue>,
}

impl<'a> BlockGroupingCollector<'a> {
    /// `new BlockGroupingCollector(groupSort, topNGroups, needsScores,
    /// lastDocPerGroup)`.
    ///
    /// # Errors
    /// `topNGroups must be >= 1` ([`Error::IllegalArgument`]).
    pub fn new(
        group_sort: Sort,
        top_n_groups: usize,
        needs_scores: bool,
        last_doc_per_group: BooleanQuery,
    ) -> Result<Self> {
        if top_n_groups < 1 {
            return Err(Error::IllegalArgument(format!(
                "topNGroups must be >= 1 (got {top_n_groups})"
            )));
        }
        Ok(Self {
            pending_sub_docs: Vec::new(),
            pending_sub_scores: Vec::new(),
            reversed: group_sort.reversed(),
            sort: group_sort,
            top_n_groups,
            last_doc_per_group,
            needs_scores,
            slots: vec![Vec::new(); top_n_groups],
            bottom_slot: 0,
            queue_full: false,
            current: None,
            top_group_doc: 0,
            total_hit_count: 0,
            total_group_count: 0,
            doc_base: 0,
            group_end_doc_id: -1,
            last_doc_per_group_bits: None,
            queue: Vec::new(),
            queue_top: 0,
            group_competes: false,
            keys: None,
            scratch: Vec::new(),
        })
    }

    /// `GroupQueue.lessThan(a, b)`: `a` is the less competitive group --
    /// it sorts after `b`, or ties and starts later.
    fn less_than(&self, a: &OneGroup<'_>, b: &OneGroup<'_>) -> bool {
        match compare_all(
            &self.sort,
            &self.reversed,
            &self.slots[a.comparator_slot],
            &self.slots[b.comparator_slot],
        ) {
            Ordering::Equal => a.top_group_doc > b.top_group_doc,
            c => c == Ordering::Greater,
        }
    }

    /// The queue's top: its least competitive group (`PriorityQueue.top()`
    /// after an `add` or `updateTop`).
    fn find_top(&mut self) {
        let mut top = 0;
        for i in 1..self.queue.len() {
            if self.less_than(&self.queue[i], &self.queue[top]) {
                top = i;
            }
        }
        self.queue_top = top;
    }

    /// `processGroup()`: the block just finished enters the queue if it
    /// competes.
    fn process_group(&mut self) {
        self.total_group_count = self.total_group_count.wrapping_add(1);
        if self.group_competes {
            let Some((ord, ref current)) = self.current else {
                return;
            };
            let leaf = OpenSegment { ..*current };
            let docs = std::mem::take(&mut self.pending_sub_docs);
            let scores = std::mem::take(&mut self.pending_sub_scores);
            let top_group_doc = self.doc_base.saturating_add(self.top_group_doc);
            if !self.queue_full {
                self.queue.push(OneGroup {
                    leaf,
                    ord,
                    top_group_doc,
                    docs,
                    scores,
                    comparator_slot: self.bottom_slot,
                });
                self.find_top();
                self.queue_full = self.queue.len() == self.top_n_groups;
                if self.queue_full {
                    self.bottom_slot = self.queue[self.queue_top].comparator_slot;
                } else {
                    self.bottom_slot = self.queue.len();
                }
            } else {
                let top = &mut self.queue[self.queue_top];
                top.leaf = leaf;
                top.ord = ord;
                top.top_group_doc = top_group_doc;
                top.docs = docs;
                top.scores = scores;
                self.find_top();
                self.bottom_slot = self.queue[self.queue_top].comparator_slot;
            }
        }
        self.pending_sub_docs.clear();
        self.pending_sub_scores.clear();
    }

    /// `getTopGroups(withinGroupSort, groupOffset, withinGroupOffset,
    /// maxDocsPerGroup)`: the queued groups from `group_offset` on, best
    /// first, each with its documents sorted by `within_group_sort`; `None`
    /// when no more than `group_offset` groups were queued.
    ///
    /// # Errors
    /// `cannot sort by relevance within group: needsScores=false`, or
    /// `maxDocsPerGroup` of `0` (both [`Error::IllegalArgument`]); what
    /// reading a sort key reports.
    pub fn top_groups(
        &self,
        within_group_sort: &Sort,
        group_offset: usize,
        within_group_offset: usize,
        max_docs_per_group: usize,
    ) -> Result<Option<TopGroups<()>>> {
        if group_offset >= self.queue.len() {
            return Ok(None);
        }
        // The queue popped least competitive first: its order, best first.
        let mut order: Vec<usize> = (0..self.queue.len()).collect();
        order.sort_by(|&a, &b| {
            if self.less_than(&self.queue[a], &self.queue[b]) {
                Ordering::Greater
            } else if self.less_than(&self.queue[b], &self.queue[a]) {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        });
        let by_relevance = within_group_sort.is_relevance();
        let group_sort_by_relevance = self.sort.is_relevance();
        let mut max_score = FLOAT_MIN_VALUE;
        let mut total_grouped_hit_count = 0i32;
        let mut groups = Vec::new();
        for &g in order.iter().skip(group_offset) {
            let og = &self.queue[g];
            if by_relevance && !self.needs_scores {
                return Err(Error::IllegalArgument(
                    "cannot sort by relevance within group: needsScores=false".into(),
                ));
            }
            let mut collector =
                GroupTopDocs::new(by_relevance, within_group_sort, max_docs_per_group, false)?;
            let mut group_max_score = if self.needs_scores {
                f32::NEG_INFINITY
            } else {
                f32::NAN
            };
            collector.set_next_reader(og.ord, &og.leaf)?;
            for (i, &doc) in og.docs.iter().enumerate() {
                let mut score = 0.0;
                if self.needs_scores {
                    score = og.scores.get(i).copied().unwrap_or(0.0);
                    if !by_relevance {
                        group_max_score = java_max_f32(group_max_score, score);
                    }
                }
                collector.collect(doc, score)?;
            }
            let count = i32::try_from(og.docs.len()).unwrap_or(i32::MAX);
            total_grouped_hit_count = total_grouped_hit_count.wrapping_add(count);
            let score_docs = collector.top_docs(within_group_offset, max_docs_per_group);
            if by_relevance {
                if let Some(first) = score_docs.first() {
                    group_max_score = first.score;
                }
            }
            if !group_sort_by_relevance {
                max_score = java_max_f32(max_score, group_max_score);
            }
            groups.push(GroupDocs {
                score: f32::NAN,
                max_score: group_max_score,
                total_hits: TotalHits {
                    value: u64::try_from(og.docs.len()).unwrap_or(u64::MAX),
                    relation: TotalHitsRelation::EqualTo,
                },
                score_docs,
                group_value: None,
                group_sort_values: self.slots[og.comparator_slot].clone(),
            });
        }
        if group_sort_by_relevance {
            max_score = groups.first().map_or(f32::NAN, |g| g.max_score);
        }
        Ok(Some(TopGroups {
            total_hit_count: self.total_hit_count,
            total_grouped_hit_count,
            total_group_count: Some(self.total_group_count),
            groups,
            group_sort: self.sort.fields.clone(),
            within_group_sort: within_group_sort.fields.clone(),
            max_score,
        }))
    }

    /// Whether the document's values (read into `scratch`) beat the bottom
    /// slot's.
    fn beats_bottom(&mut self, doc: i32, score: f32) -> Result<bool> {
        let keys = self
            .keys
            .as_mut()
            .ok_or_else(|| Error::IllegalState("collected before entering a segment".into()))?;
        if keys.compare_doc(
            &self.sort,
            &self.reversed,
            &self.slots[self.bottom_slot],
            doc,
            score,
        )? != Ordering::Greater
        {
            return Ok(false);
        }
        keys.values_into(&self.sort, doc, score, &mut self.scratch)?;
        Ok(true)
    }
}

impl<'a> SegmentCollector<'a> for BlockGroupingCollector<'a> {
    fn score_mode(&self) -> ScoreMode {
        if self.needs_scores {
            ScoreMode::Complete
        } else {
            ScoreMode::CompleteNoScores
        }
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.pending_sub_docs.clear();
        self.pending_sub_scores.clear();
        self.doc_base = leaf.doc_base;
        self.last_doc_per_group_bits = crate::join::query_bit_set(leaf, &self.last_doc_per_group)?;
        self.group_end_doc_id = -1;
        self.current = Some((ord, OpenSegment { ..*leaf }));
        self.keys = Some(LeafKeys::open(&self.sort, leaf)?);
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        if doc > self.group_end_doc_id {
            if !self.pending_sub_docs.is_empty() {
                self.process_group();
            }
            let bits = self.last_doc_per_group_bits.as_ref().ok_or_else(|| {
                // Java dereferences the segment's missing scorer here.
                Error::IllegalState(
                    "a segment with hits has no document closing a group (lastDocPerGroup)".into(),
                )
            })?;
            self.group_end_doc_id = usize::try_from(doc)
                .ok()
                .and_then(|d| bits.next_set_bit(d))
                .and_then(|d| i32::try_from(d).ok())
                .unwrap_or(crate::exec::NO_MORE_DOCS);
            self.pending_sub_docs.clear();
            self.pending_sub_scores.clear();
            self.group_competes = !self.queue_full;
        }
        self.total_hit_count = self.total_hit_count.wrapping_add(1);
        self.pending_sub_docs.push(doc);
        if self.needs_scores {
            self.pending_sub_scores.push(score);
        }
        let first = self.pending_sub_docs.len() == 1;
        if self.group_competes && first {
            let keys = self
                .keys
                .as_mut()
                .ok_or_else(|| Error::IllegalState("collected before entering a segment".into()))?;
            keys.values_into(&self.sort, doc, score, &mut self.slots[self.bottom_slot])?;
            self.top_group_doc = doc;
            return Ok(());
        }
        // Either the group already competes and the document must beat its
        // best so far, or the queue is full and it must beat the bottom
        // group: both are the bottom slot.
        if !self.beats_bottom(doc, score)? {
            return Ok(());
        }
        self.group_competes = true;
        let slot = &mut self.slots[self.bottom_slot];
        slot.clear();
        slot.extend_from_slice(&self.scratch);
        self.top_group_doc = doc;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if !self.pending_sub_docs.is_empty() {
            self.process_group();
        }
        Ok(())
    }
}

/// `BlockGroupingCollectorManager<T>`: a [`BlockGroupingCollector`] of
/// `groupOffset + topNGroups` groups per slice, merged by
/// [`TopGroups::merge_block_groups`].
pub struct BlockGroupingCollectorManager {
    group_sort: Sort,
    group_offset: usize,
    top_n_groups: usize,
    needs_scores: bool,
    last_doc_per_group: BooleanQuery,
    within_group_sort: Sort,
    within_group_offset: usize,
    max_docs_per_group: usize,
}

impl BlockGroupingCollectorManager {
    /// `new BlockGroupingCollectorManager(groupSort, groupOffset,
    /// topNGroups, needsScores, lastDocPerGroup, withinGroupSort,
    /// withinGroupOffset, maxDocsPerGroup)`.
    ///
    /// # Errors
    /// Java's `IllegalArgumentException`s: `topNGroups must be >= 1`,
    /// `maxDocsPerGroup must be >= 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        group_sort: Sort,
        group_offset: usize,
        top_n_groups: usize,
        needs_scores: bool,
        last_doc_per_group: BooleanQuery,
        within_group_sort: Sort,
        within_group_offset: usize,
        max_docs_per_group: usize,
    ) -> Result<Self> {
        if top_n_groups < 1 {
            return Err(Error::IllegalArgument(format!(
                "topNGroups must be >= 1 (got {top_n_groups})"
            )));
        }
        if max_docs_per_group < 1 {
            return Err(Error::IllegalArgument(format!(
                "maxDocsPerGroup must be >= 1 (got {max_docs_per_group})"
            )));
        }
        Ok(Self {
            group_sort,
            group_offset,
            top_n_groups,
            needs_scores,
            last_doc_per_group,
            within_group_sort,
            within_group_offset,
            max_docs_per_group,
        })
    }
}

impl<'a> GroupingCollectorManager<'a> for BlockGroupingCollectorManager {
    type Collector = BlockGroupingCollector<'a>;
    type Output = TopGroups<()>;

    fn new_collector(&self) -> Result<Self::Collector> {
        BlockGroupingCollector::new(
            self.group_sort.clone(),
            self.group_offset.saturating_add(self.top_n_groups),
            self.needs_scores,
            self.last_doc_per_group.clone(),
        )
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        let mut shards = Vec::new();
        for c in &collectors {
            if let Some(tg) = c.top_groups(
                &self.within_group_sort,
                0,
                self.within_group_offset,
                self.max_docs_per_group,
            )? {
                if !tg.groups.is_empty() {
                    shards.push(tg);
                }
            }
        }
        Ok(TopGroups::merge_block_groups(
            &shards,
            &self.group_sort,
            self.group_offset,
            self.top_n_groups,
            &self.within_group_sort,
        ))
    }
}
