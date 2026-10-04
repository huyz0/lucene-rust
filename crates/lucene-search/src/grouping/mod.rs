//! `lucene-grouping` (Lucene 10.5.0, `org.apache.lucene.search.grouping`):
//! grouping a search's hits by a value -- a keyword field's term
//! ([`TermGroupSelector`]), a numeric range ([`LongRangeGroupSelector`],
//! [`DoubleRangeGroupSelector`]) or a document block
//! ([`BlockGroupingCollector`]) -- and returning the top groups, each with
//! its top documents.
//!
//! - Two passes, as Java runs them: [`FirstPassGroupingCollector`] finds the
//!   top groups by a group sort ([`SearchGroup`]s), then
//!   [`TopGroupsCollector`] (a [`SecondPassGroupingCollector`] over a
//!   per-group top-docs [`GroupReducer`]) collects each one's top documents
//!   ([`TopGroups`], [`GroupDocs`]). [`GroupingSearch`] drives both, with
//!   [`AllGroupsCollector`] (the number of groups) and
//!   [`AllGroupHeadsCollector`] (each group's best document), and the
//!   collector managers merge per-slice results as Java's do
//!   ([`SearchGroup::merge`], [`TopGroups::merge`],
//!   [`TopGroups::merge_block_groups`]).
//! - [`DistinctValuesCollector`]: the distinct values of a second selector
//!   per group; [`TermGroupFacetCollector`]: facet counts that count each
//!   group once.
//!
//! Every collector is a [`SegmentCollector`], reading each segment's doc
//! values as the search reaches it, and a manager runs one per slice
//! ([`search_manager`]); sorts are [`Sort`]s of
//! [`crate::top_field::SortField`] keys.
//!
//! # What differs from Java
//!
//! - A group value is an `Option` (`null`: the documents without one), and
//!   collections Java keeps in hash sets and hash maps (`getGroups()`,
//!   `retrieveGroupHeads()`, `GroupCount.uniqueValues`) come back as vectors
//!   in first-seen order (Java's order is the hash's, i.e. unspecified).
//! - `GroupSelector.setScorer` is folded into
//!   [`GroupSelector::advance_to`], which is handed the document's score.
//! - Comparator slots hold the values `FieldComparator.value(slot)` returns
//!   ([`GroupSortValue`]) and compare them with `compareValues`, which orders
//!   as Java's slot comparisons do (see [`sort`]).
//! - Results are read without consuming the collectors: Java's
//!   `getTopGroups` pops its queues, so a second call there returns less.
//! - A selector Java shares between collectors (`GroupingSearch`'s first
//!   round, `DistinctValuesCollector`'s value selector) is one per collector
//!   here, or owned once by the collector: their state is a cache of the
//!   segment's ordinals, so the groups found are the same.
//! - `ValueSourceGroupSelector` (and `GroupingSearch`'s `ValueSource`
//!   constructor) needs `lucene-queries`' `ValueSource`/`FunctionValues`
//!   (`MutableValue` group values), which are M10's T10.5: not ported yet.

mod block;
mod collectors;
mod facet;
mod search;
mod selector;
pub mod sort;

pub use block::{BlockGroupingCollector, BlockGroupingCollectorManager};
pub use collectors::{
    AllGroupHeadsCollector, AllGroupHeadsCollectorManager, AllGroupsCollector,
    AllGroupsCollectorManager, CollectorsReducer, DistinctValuesCollector,
    DistinctValuesCollectorManager, DistinctValuesReducer, FirstPassGroupingCollector,
    FirstPassGroupingCollectorManager, GroupCount, GroupHead, GroupHeadsResult, GroupTopDocs,
    SecondPassGroupingCollector, TopGroupsCollector, TopGroupsCollectorManager,
};
pub use facet::{FacetEntry, GroupedFacetResult, TermGroupFacetCollector};
pub use search::{GroupingSearch, GroupingSearchResult};
pub use selector::{
    DoubleRange, DoubleRangeFactory, DoubleRangeGroupSelector, LongRange, LongRangeFactory,
    LongRangeGroupSelector, TermGroupSelector,
};
pub use sort::{GroupSortValue, Sort};

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use crate::collector::{TotalHits, TotalHitsRelation};
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::{search_slices, SegmentCollector};
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::top_field::SortField;
use crate::{Error, Result};

/// `GroupSelector.State`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupState {
    Skip,
    Accept,
}

/// `GroupSelector<T>`: what group a document belongs to.
pub trait GroupSelector<'a> {
    /// The group value (`T`); a document without one is in the `None`
    /// group.
    type Value: Clone + Eq + Hash + fmt::Debug;
    /// `setNextReader(readerContext)`: segment `ord` of the searcher.
    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()>;
    /// `setScorer(scorer)`, which a collector calls after entering a
    /// segment -- or, as Java's `AllGroupsCollector`,
    /// `AllGroupHeadsCollector` and a distinct-values collector's value
    /// selector, never does. The range selectors open their values source
    /// here; without it they have none, and fail as Java's do
    /// (`NullPointerException`, here [`Error::IllegalState`]).
    fn set_scorer(&mut self) -> Result<()> {
        Ok(())
    }
    /// `setScorer` + `advanceTo(doc)`: positions on segment document `doc`
    /// (scoring `score`) and says whether its group is one to collect.
    fn advance_to(&mut self, doc: i32, score: f32) -> Result<GroupState>;
    /// `currentValue()`.
    fn current_value(&self) -> Option<&Self::Value>;
    /// `copyValue()`.
    fn copy_value(&self) -> Option<Self::Value> {
        self.current_value().cloned()
    }
    /// `setGroups(searchGroups)`: from now on only these groups (and, when
    /// one of them is `None`, documents without a value) are accepted.
    fn set_groups(&mut self, groups: &[SearchGroup<Self::Value>]);
}

/// `SearchGroup<T>`: a group and the values of its group sort.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchGroup<V> {
    pub group_value: Option<V>,
    pub sort_values: Vec<GroupSortValue>,
}

/// `CollectedSearchGroup<T>`: a group the first pass holds -- its value,
/// its best document (global id) and the comparator slot of that
/// document's sort values.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedSearchGroup<V> {
    pub group_value: Option<V>,
    pub top_doc: i32,
    pub comparator_slot: usize,
}

/// The multiply-rotate hash rustc uses (`FxHasher`): the group maps hash a
/// group value per collected document, where SipHash's DoS resistance buys
/// nothing (stage 3: SipHash was a tenth of a term grouping's time).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

impl std::hash::Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let mut w = [0u8; 8];
            w.copy_from_slice(c);
            self.add(u64::from_le_bytes(w));
        }
        for &b in chunks.remainder() {
            self.add(u64::from(b));
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

impl FxHasher {
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

/// A `HashMap` under [`FxHasher`].
pub(crate) type FxHashMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;

/// A map from group values (`null` included) to an index, looked up by
/// reference -- Java's `HashMap<T, ...>` with its one `null` key.
#[derive(Debug, Clone)]
pub(crate) struct GroupIndex<V> {
    map: FxHashMap<V, usize>,
    null: Option<usize>,
}

impl<V: Eq + Hash> Default for GroupIndex<V> {
    fn default() -> Self {
        Self {
            map: FxHashMap::default(),
            null: None,
        }
    }
}

impl<V: Eq + Hash> GroupIndex<V> {
    pub(crate) fn get(&self, v: Option<&V>) -> Option<usize> {
        match v {
            None => self.null,
            Some(v) => self.map.get(v).copied(),
        }
    }

    pub(crate) fn insert(&mut self, v: Option<V>, i: usize) {
        match v {
            None => self.null = Some(i),
            Some(v) => {
                self.map.insert(v, i);
            }
        }
    }

    pub(crate) fn remove(&mut self, v: Option<&V>) {
        match v {
            None => self.null = None,
            Some(v) => {
                self.map.remove(v);
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.map
            .len()
            .saturating_add(usize::from(self.null.is_some()))
    }
}

/// `GroupReducer<T, C>`: what the second pass hands each document of a top
/// group to. Java's is a map from group value to a `Collector` made by
/// `newCollector()` ([`CollectorsReducer`]); the trait lets a reducer keep
/// its per-group state its own way ([`DistinctValuesReducer`]).
pub trait GroupReducer<'a, V> {
    /// `setGroups(groups)`: one collector per group.
    fn set_groups(&mut self, groups: &[SearchGroup<V>]);
    /// `needsScores()`.
    fn needs_scores(&self) -> bool;
    /// `setNextReader(ctx)`: every group's collector enters segment `ord`.
    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()>;
    /// `collect(value, doc)`: segment document `doc` (scoring `score`) of
    /// the group `value`.
    fn collect(&mut self, value: Option<&V>, doc: i32, score: f32) -> Result<()>;
}

/// `CollectorManager<C, T>` for a [`SegmentCollector`]: one collector per
/// slice of the searcher, reduced to a result.
pub trait GroupingCollectorManager<'a> {
    type Collector: SegmentCollector<'a>;
    type Output;
    /// `newCollector()`.
    fn new_collector(&self) -> Result<Self::Collector>;
    /// `reduce(collectors)`, the collectors in slice order.
    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output>;
}

/// `searcher.search(query, manager)`: a collector per slice (in order, on
/// this thread), then the manager's `reduce`.
///
/// # Errors
/// What the search, a collector or the reduction reports.
pub fn search_manager<'a, M: GroupingCollectorManager<'a>>(
    searcher: &IndexSearcher<'_, 'a>,
    query: &BooleanQuery,
    manager: &M,
) -> Result<M::Output> {
    let collectors = search_slices(searcher, query, || manager.new_collector())?;
    manager.reduce(collectors)
}

/// One hit of a group: `ScoreDoc`, or a `FieldDoc` with its sort values.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupScoreDoc {
    /// The global document id.
    pub doc: i32,
    /// The score; `NaN` for a hit of a field sort (`TopFieldCollector`
    /// keeps none).
    pub score: f32,
    /// `FieldDoc.fields`; `None` for a hit sorted by relevance.
    pub fields: Option<Vec<GroupSortValue>>,
    /// `ScoreDoc.shardIndex`: set by a merge, `-1` before.
    pub shard_index: i32,
}

/// `GroupDocs<T>`: one group's hits.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupDocs<V> {
    /// The group's score (`NaN` unless a merge combined one).
    pub score: f32,
    pub max_score: f32,
    pub total_hits: TotalHits,
    pub score_docs: Vec<GroupScoreDoc>,
    /// The group value; `None` too for a block group.
    pub group_value: Option<V>,
    pub group_sort_values: Vec<GroupSortValue>,
}

/// `TopGroups<T>`: the top groups with their hits.
#[derive(Debug, Clone, PartialEq)]
pub struct TopGroups<V> {
    pub total_hit_count: i32,
    pub total_grouped_hit_count: i32,
    /// The number of groups, when it was counted.
    pub total_group_count: Option<i32>,
    pub groups: Vec<GroupDocs<V>>,
    pub group_sort: Vec<SortField>,
    pub within_group_sort: Vec<SortField>,
    pub max_score: f32,
}

/// `TopGroups.ScoreMergeMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreMergeMode {
    None,
    /// The sum of the shards' group scores.
    Total,
    /// Their sum over the group's hits.
    Avg,
}

/// `TopGroups.nonNANmax`.
pub(crate) fn non_nan_max(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        crate::join::query_time::java_max_f32(a, b)
    }
}

/// A `TreeSet` under a comparator that the elements' state decides: kept
/// sorted, an element comparing equal to one already in is not added, and
/// `remove` takes out the element comparing equal (as Java's does) -- both
/// faithful to `TreeSet`, including where a comparator that is not a total
/// order makes it drop elements.
#[derive(Debug, Default, Clone)]
pub(crate) struct TreeSet<T> {
    items: Vec<T>,
}

impl<T: Copy + PartialEq> TreeSet<T> {
    pub(crate) fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn search(
        &self,
        x: &T,
        cmp: &mut dyn FnMut(&T, &T) -> Ordering,
    ) -> std::result::Result<usize, usize> {
        self.items.binary_search_by(|e| cmp(e, x))
    }

    /// `add(x)`: whether it was added.
    pub(crate) fn add(&mut self, x: T, cmp: &mut dyn FnMut(&T, &T) -> Ordering) -> bool {
        match self.search(&x, cmp) {
            Ok(_) => false,
            Err(i) => {
                self.items.insert(i, x);
                true
            }
        }
    }

    /// `remove(x)`: the element comparing equal to `x`.
    pub(crate) fn remove(&mut self, x: &T, cmp: &mut dyn FnMut(&T, &T) -> Ordering) -> bool {
        match self.search(x, cmp) {
            Ok(i) => {
                self.items.remove(i);
                true
            }
            Err(_) => false,
        }
    }

    pub(crate) fn first(&self) -> Option<T> {
        self.items.first().copied()
    }

    pub(crate) fn last(&self) -> Option<T> {
        self.items.last().copied()
    }

    pub(crate) fn poll_first(&mut self) -> Option<T> {
        if self.items.is_empty() {
            None
        } else {
            Some(self.items.remove(0))
        }
    }

    pub(crate) fn poll_last(&mut self) -> Option<T> {
        self.items.pop()
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
}

impl<V: Clone + Eq + Hash> SearchGroup<V> {
    /// `SearchGroup.merge(topGroups, offset, topN, groupSort)`: the top
    /// groups of several shards' (or slices') first passes, each group's
    /// sort values the best any shard reported, ties to the lower shard.
    pub fn merge(
        top_groups: &[Vec<SearchGroup<V>>],
        offset: usize,
        top_n: usize,
        group_sort: &Sort,
    ) -> Vec<SearchGroup<V>> {
        GroupMerger::new(group_sort, top_groups).merge(offset, top_n)
    }
}

/// `SearchGroup.MergedGroup`.
struct MergedGroup<V> {
    group_value: Option<V>,
    top_values: Vec<GroupSortValue>,
    shards: Vec<usize>,
    min_shard_index: usize,
    processed: bool,
    in_queue: bool,
}

/// `SearchGroup.GroupMerger`.
struct GroupMerger<'s, V> {
    sort: &'s Sort,
    reversed: Vec<i32>,
    shards: &'s [Vec<SearchGroup<V>>],
    /// Each `ShardIter`'s position.
    pos: Vec<usize>,
    merged: Vec<MergedGroup<V>>,
    seen: HashMap<Option<V>, usize>,
    queue: TreeSet<usize>,
}

impl<'s, V: Clone + Eq + Hash> GroupMerger<'s, V> {
    fn new(sort: &'s Sort, shards: &'s [Vec<SearchGroup<V>>]) -> Self {
        Self {
            sort,
            reversed: sort.reversed(),
            shards,
            pos: vec![0; shards.len()],
            merged: Vec::new(),
            seen: HashMap::new(),
            queue: TreeSet::new(),
        }
    }

    /// `GroupComparator.compare`.
    fn cmp(
        sort: &Sort,
        reversed: &[i32],
        merged: &[MergedGroup<V>],
        a: usize,
        b: usize,
    ) -> Ordering {
        if a == b {
            return Ordering::Equal;
        }
        let (ga, gb) = (&merged[a], &merged[b]);
        match sort::compare_all(sort, reversed, &ga.top_values, &gb.top_values) {
            Ordering::Equal => ga.min_shard_index.cmp(&gb.min_shard_index),
            c => c,
        }
    }

    fn update_next_group(&mut self, top_n: usize, shard: usize) {
        let (sort, reversed) = (self.sort, self.reversed.clone());
        while let Some(group) = self.shards[shard].get(self.pos[shard]) {
            self.pos[shard] = self.pos[shard].saturating_add(1);
            let m = match self.seen.get(&group.group_value) {
                None => {
                    let m = self.merged.len();
                    self.merged.push(MergedGroup {
                        group_value: group.group_value.clone(),
                        top_values: group.sort_values.clone(),
                        shards: Vec::new(),
                        min_shard_index: shard,
                        processed: false,
                        in_queue: true,
                    });
                    self.seen.insert(group.group_value.clone(), m);
                    let merged = &self.merged;
                    self.queue
                        .add(m, &mut |a, b| Self::cmp(sort, &reversed, merged, *a, *b));
                    m
                }
                Some(&m) if self.merged[m].processed => continue,
                Some(&m) => {
                    let current = &self.merged[m];
                    let competes = match sort::compare_all(
                        sort,
                        &reversed,
                        &group.sort_values,
                        &current.top_values,
                    ) {
                        Ordering::Less => true,
                        Ordering::Greater => false,
                        Ordering::Equal => shard < current.min_shard_index,
                    };
                    if competes {
                        let skip_heavy_ops = self.queue.first() == Some(m);
                        if self.merged[m].in_queue && !skip_heavy_ops {
                            let merged = &self.merged;
                            self.queue
                                .remove(&m, &mut |a, b| Self::cmp(sort, &reversed, merged, *a, *b));
                        }
                        self.merged[m].top_values = group.sort_values.clone();
                        self.merged[m].min_shard_index = shard;
                        if !skip_heavy_ops {
                            let merged = &self.merged;
                            self.queue
                                .add(m, &mut |a, b| Self::cmp(sort, &reversed, merged, *a, *b));
                        }
                        self.merged[m].in_queue = true;
                    }
                    m
                }
            };
            self.merged[m].shards.push(shard);
            break;
        }
        while self.queue.len() > top_n {
            if let Some(g) = self.queue.poll_last() {
                self.merged[g].in_queue = false;
            }
        }
    }

    fn merge(mut self, offset: usize, top_n: usize) -> Vec<SearchGroup<V>> {
        let max_queue_size = offset.saturating_add(top_n);
        for shard in 0..self.shards.len() {
            if !self.shards[shard].is_empty() {
                self.update_next_group(max_queue_size, shard);
            }
        }
        let mut out = Vec::new();
        let mut count = 0usize;
        while let Some(g) = self.queue.poll_first() {
            self.merged[g].processed = true;
            let skip = count < offset;
            count = count.saturating_add(1);
            if !skip {
                out.push(SearchGroup {
                    group_value: self.merged[g].group_value.clone(),
                    sort_values: self.merged[g].top_values.clone(),
                });
                if out.len() == top_n {
                    break;
                }
            }
            for shard in self.merged[g].shards.clone() {
                self.update_next_group(max_queue_size, shard);
            }
        }
        out
    }
}

/// `TopDocs.merge(topN, shardHits)` / `TopDocs.merge(sort, topN,
/// shardHits)`: a k-way merge of the shards' hits, each shard's taken in its
/// own order, the next hit the best of the shards' heads -- by score
/// (highest first) or by `sort` -- ties to the lower shard, then the lower
/// document.
fn merge_hits(
    doc_sort: Option<&Sort>,
    top_n: usize,
    shards: Vec<Vec<GroupScoreDoc>>,
) -> Vec<GroupScoreDoc> {
    let reversed = doc_sort.map(Sort::reversed).unwrap_or_default();
    // `MergeSortQueue.lessThan`/`ScoreMergeSortQueue.lessThan`: whether `a`
    // comes before `b`.
    let before = |a: &GroupScoreDoc, b: &GroupScoreDoc| -> bool {
        let primary = match doc_sort {
            // `first.score < second.score` is "after", `>` is "before";
            // anything else (equal, or `NaN`) goes to the tie-breaker.
            None => {
                if a.score < b.score {
                    Ordering::Greater
                } else if a.score > b.score {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            }
            Some(sort) => sort::compare_all(
                sort,
                &reversed,
                a.fields.as_deref().unwrap_or(&[]),
                b.fields.as_deref().unwrap_or(&[]),
            ),
        };
        primary
            .then(a.shard_index.cmp(&b.shard_index))
            .then(a.doc.cmp(&b.doc))
            == Ordering::Less
    };
    let mut at = vec![0usize; shards.len()];
    let mut out = Vec::new();
    while out.len() < top_n {
        let mut best: Option<usize> = None;
        for (s, hits) in shards.iter().enumerate() {
            let Some(h) = hits.get(at[s]) else {
                continue;
            };
            if best.is_none_or(|b| before(h, &shards[b][at[b]])) {
                best = Some(s);
            }
        }
        let Some(b) = best else {
            break;
        };
        out.push(shards[b][at[b]].clone());
        at[b] = at[b].saturating_add(1);
    }
    out
}

impl<V: Clone + PartialEq> TopGroups<V> {
    /// `new TopGroups(oldTopGroups, totalGroupCount)`.
    pub fn with_total_group_count(mut self, total_group_count: Option<i32>) -> Self {
        self.total_group_count = total_group_count;
        self
    }

    /// `TopGroups.merge(shardGroups, groupSort, docSort, docOffset, docTopN,
    /// scoreMergeMode)`: the second passes of several shards (or slices)
    /// over the same groups, merged group by group; `None` for no shards.
    ///
    /// # Errors
    /// Java's `IllegalArgumentException`s when the shards disagree on the
    /// groups.
    pub fn merge(
        shard_groups: &[TopGroups<V>],
        group_sort: &Sort,
        doc_sort: &Sort,
        doc_offset: usize,
        doc_top_n: usize,
        score_merge_mode: ScoreMergeMode,
    ) -> Result<Option<TopGroups<V>>> {
        let Some(first) = shard_groups.first() else {
            return Ok(None);
        };
        let mut total_hit_count = 0i32;
        let mut total_grouped_hit_count = 0i32;
        let mut total_group_count: Option<i32> = None;
        let num_groups = first.groups.len();
        for shard in shard_groups {
            if shard.groups.len() != num_groups {
                return Err(Error::IllegalArgument(
                    "number of groups differs across shards; you must pass same top groups to \
                     all shards' second-pass collector"
                        .into(),
                ));
            }
            total_hit_count = total_hit_count.wrapping_add(shard.total_hit_count);
            total_grouped_hit_count =
                total_grouped_hit_count.wrapping_add(shard.total_grouped_hit_count);
            if let Some(n) = shard.total_group_count {
                total_group_count = Some(total_group_count.unwrap_or(0).wrapping_add(n));
            }
        }
        let sort_by_relevance = doc_sort.is_relevance();
        let mut merged_groups = Vec::with_capacity(num_groups);
        let mut total_max_score = f32::NAN;
        for g in 0..num_groups {
            let group_value = first.groups[g].group_value.clone();
            let mut max_score = f32::NAN;
            let mut total_hits = 0i64;
            let mut score_sum = 0.0f64;
            let mut shard_hits = Vec::with_capacity(shard_groups.len());
            for (shard_idx, shard) in shard_groups.iter().enumerate() {
                let docs = &shard.groups[g];
                if docs.group_value != group_value {
                    return Err(Error::IllegalArgument(
                        "group values differ across shards; you must pass same top groups to \
                         all shards' second-pass collector"
                            .into(),
                    ));
                }
                let mut hits = docs.score_docs.clone();
                for h in &mut hits {
                    h.shard_index = i32::try_from(shard_idx).unwrap_or(i32::MAX);
                }
                shard_hits.push(hits);
                if !sort_by_relevance {
                    max_score = non_nan_max(max_score, docs.max_score);
                }
                total_hits = total_hits.wrapping_add(docs.total_hits.value as i64);
                score_sum += f64::from(docs.score);
            }
            let merged = merge_hits(
                (!sort_by_relevance).then_some(doc_sort),
                doc_offset.saturating_add(doc_top_n),
                shard_hits,
            );
            if sort_by_relevance {
                max_score = merged.first().map_or(f32::NAN, |h| h.score);
            }
            let score_docs: Vec<GroupScoreDoc> = merged.into_iter().skip(doc_offset).collect();
            let group_score = match score_merge_mode {
                ScoreMergeMode::None => f32::NAN,
                ScoreMergeMode::Avg if total_hits > 0 => (score_sum / total_hits as f64) as f32,
                ScoreMergeMode::Avg => f32::NAN,
                ScoreMergeMode::Total => score_sum as f32,
            };
            merged_groups.push(GroupDocs {
                score: group_score,
                max_score,
                total_hits: TotalHits {
                    value: u64::try_from(total_hits).unwrap_or(0),
                    relation: TotalHitsRelation::EqualTo,
                },
                score_docs,
                group_value,
                group_sort_values: first.groups[g].group_sort_values.clone(),
            });
            total_max_score = non_nan_max(total_max_score, max_score);
        }
        Ok(Some(TopGroups {
            total_hit_count,
            total_grouped_hit_count,
            total_group_count,
            groups: merged_groups,
            group_sort: group_sort.fields.clone(),
            within_group_sort: doc_sort.fields.clone(),
            max_score: total_max_score,
        }))
    }

    /// `TopGroups.mergeBlockGroups(shardGroups, groupSort, groupOffset,
    /// topNGroups, docSort)`: block groups of several slices, merged by
    /// their group sort values (ties to the lower slice).
    pub fn merge_block_groups(
        shard_groups: &[TopGroups<V>],
        group_sort: &Sort,
        group_offset: usize,
        top_n_groups: usize,
        doc_sort: &Sort,
    ) -> TopGroups<V> {
        let mut total_group_count: Option<i32> = None;
        let mut total_hit_count = 0i32;
        let mut total_grouped_hit_count = 0i32;
        for sg in shard_groups {
            total_hit_count = total_hit_count.wrapping_add(sg.total_hit_count);
            if let Some(n) = sg.total_group_count {
                total_group_count = Some(total_group_count.unwrap_or(0).wrapping_add(n));
            }
        }
        let reversed = group_sort.reversed();
        // `MergedBlockGroup(topValues, shardIndex, groupIndex)`.
        let cmp = |a: &(usize, usize), b: &(usize, usize)| -> Ordering {
            if a == b {
                return Ordering::Equal;
            }
            let va = &shard_groups[a.0].groups[a.1].group_sort_values;
            let vb = &shard_groups[b.0].groups[b.1].group_sort_values;
            match sort::compare_all(group_sort, &reversed, va, vb) {
                Ordering::Equal => a.0.cmp(&b.0),
                c => c,
            }
        };
        let mut queue: TreeSet<(usize, usize)> = TreeSet::new();
        let mut total_max_score = f32::NAN;
        let group_sort_by_relevance = group_sort.is_relevance();
        for (idx, tg) in shard_groups.iter().enumerate() {
            if tg.groups.is_empty() {
                continue;
            }
            if !group_sort_by_relevance {
                total_max_score = non_nan_max(total_max_score, tg.max_score);
            }
            queue.add((idx, 0), &mut |a, b| cmp(a, b));
        }
        if group_sort_by_relevance {
            if let Some((shard, _)) = queue.first() {
                total_max_score = shard_groups[shard].max_score;
            }
        }
        let mut groups = Vec::new();
        let mut count = 0usize;
        while let Some((shard, gi)) = queue.poll_first() {
            let docs = &shard_groups[shard].groups[gi];
            let skip = count < group_offset;
            count = count.saturating_add(1);
            if !skip {
                groups.push(docs.clone());
                total_grouped_hit_count =
                    total_grouped_hit_count.wrapping_add(docs.total_hits.value as i32);
                if groups.len() == top_n_groups {
                    break;
                }
            }
            let next = gi.saturating_add(1);
            if next < shard_groups[shard].groups.len() {
                queue.add((shard, next), &mut |a, b| cmp(a, b));
            }
        }
        TopGroups {
            total_hit_count,
            total_grouped_hit_count,
            total_group_count,
            groups,
            group_sort: group_sort.fields.clone(),
            within_group_sort: doc_sort.fields.clone(),
            max_score: total_max_score,
        }
    }
}

#[cfg(test)]
mod tests;
