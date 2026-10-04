//! The grouping collectors over a [`GroupSelector`]: the first pass
//! (`FirstPassGroupingCollector`), the second pass
//! (`SecondPassGroupingCollector` with its `GroupReducer`s:
//! `TopGroupsCollector`, `DistinctValuesCollector`), `AllGroupsCollector`,
//! `AllGroupHeadsCollector`, and their collector managers.

use std::cmp::Ordering;
use std::hash::Hash;
use std::marker::PhantomData;

use lucene_util::fixed_bit_set::FixedBitSet;

use super::sort::{compare_all, compare_values, GroupSortValue, LeafKeys, Sort};
use super::{
    non_nan_max, CollectedSearchGroup, GroupDocs, GroupIndex, GroupReducer, GroupScoreDoc,
    GroupSelector, GroupState, GroupingCollectorManager, ScoreMergeMode, SearchGroup, TopGroups,
    TreeSet,
};
use crate::collector::{ScoreMode, TotalHits, TotalHitsRelation};
use crate::join::query_time::java_max_f32;
use crate::leaf_collector::SegmentCollector;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

/// `Float.MIN_VALUE`: the smallest positive float, where Java's max-score
/// accumulators start.
pub(crate) const FLOAT_MIN_VALUE: f32 = f32::from_bits(1);

/// A document of a group a reducer was not given (Java's
/// `NullPointerException` in `GroupReducer.collect`).
fn not_given() -> Error {
    Error::IllegalState("a document of a group the reducer was not given".into())
}

fn no_leaf() -> Error {
    Error::IllegalState("a grouping collector collected before entering a segment".into())
}

/// Whether a document with `doc` values beats the slot holding `slot`:
/// `reverseMul * compareBottom(doc) > 0` over the keys in order (the first
/// that differs decides; all equal does not compete, as the earlier
/// document wins).
fn beats(sort: &Sort, reversed: &[i32], slot: &[GroupSortValue], doc: &[GroupSortValue]) -> bool {
    compare_all(sort, reversed, slot, doc) == Ordering::Greater
}

// ---------------------------------------------------------------------------
// First pass
// ---------------------------------------------------------------------------

/// `FirstPassGroupingCollector<T>`: the top `topNGroups` groups by
/// `groupSort`, each group sorting as its best document does.
pub struct FirstPassGroupingCollector<'a, S: GroupSelector<'a>> {
    selector: S,
    ignore_docs_without_group_field: bool,
    sort: Sort,
    reversed: Vec<i32>,
    top_n_groups: usize,
    needs_scores: bool,
    /// `groupMap`: a group value to its index in `groups`.
    group_map: GroupIndex<S::Value>,
    groups: Vec<CollectedSearchGroup<S::Value>>,
    /// The comparators' slots (`topNGroups + 1` of them, one spare).
    slots: Vec<Vec<GroupSortValue>>,
    /// `orderedGroups`, once `topNGroups` groups were seen: indices into
    /// `groups`.
    ordered: Option<TreeSet<usize>>,
    /// The slot `setBottom` last named.
    bottom_slot: usize,
    doc_base: i32,
    spare_slot: usize,
    keys: Option<LeafKeys<'a>>,
}

impl<'a, S: GroupSelector<'a>> FirstPassGroupingCollector<'a, S> {
    /// `new FirstPassGroupingCollector(groupSelector, groupSort, topNGroups,
    /// ignoreDocsWithoutGroupField)`.
    ///
    /// # Errors
    /// `topNGroups must be >= 1` ([`Error::IllegalArgument`]).
    pub fn new(
        selector: S,
        group_sort: Sort,
        top_n_groups: usize,
        ignore_docs_without_group_field: bool,
    ) -> Result<Self> {
        if top_n_groups < 1 {
            return Err(Error::IllegalArgument(format!(
                "topNGroups must be >= 1 (got {top_n_groups})"
            )));
        }
        let slot_count = top_n_groups.saturating_add(1);
        Ok(Self {
            selector,
            ignore_docs_without_group_field,
            reversed: group_sort.reversed(),
            needs_scores: group_sort.needs_scores(),
            sort: group_sort,
            top_n_groups,
            group_map: GroupIndex::default(),
            groups: Vec::new(),
            slots: vec![Vec::new(); slot_count],
            ordered: None,
            bottom_slot: 0,
            doc_base: 0,
            spare_slot: top_n_groups,
            keys: None,
        })
    }

    /// `getGroupSelector()`.
    pub fn group_selector(&self) -> &S {
        &self.selector
    }

    /// Takes the selector back (for the second pass, as Java hands the same
    /// one on).
    pub fn into_group_selector(self) -> S {
        self.selector
    }

    fn is_group_map_full(&self) -> bool {
        self.group_map.len() >= self.top_n_groups
    }

    /// `buildSortedSet()`: the groups by their slots' values, ties to the
    /// earlier top document.
    fn build_sorted_set(&mut self) {
        let mut set = TreeSet::new();
        let (sort, reversed, slots, groups) =
            (&self.sort, &self.reversed, &self.slots, &self.groups);
        for g in 0..groups.len() {
            set.add(g, &mut |a: &usize, b: &usize| {
                group_order(sort, reversed, slots, groups, *a, *b)
            });
        }
        self.ordered = Some(set);
    }

    /// `getTopGroups(groupOffset)`: the collected groups from `offset` on,
    /// best first, each with its sort values; `None` when no more than
    /// `offset` groups were collected.
    ///
    /// # Errors
    /// `groupOffset must be >= 0` cannot happen with a `usize`; none.
    pub fn top_groups(&mut self, offset: usize) -> Option<Vec<SearchGroup<S::Value>>> {
        if self.group_map.len() <= offset {
            return None;
        }
        if self.ordered.is_none() {
            self.build_sorted_set();
        }
        let ordered = self.ordered.as_ref()?;
        Some(
            ordered
                .iter()
                .skip(offset)
                .map(|&g| SearchGroup {
                    group_value: self.groups[g].group_value.clone(),
                    sort_values: self.slots[self.groups[g].comparator_slot].clone(),
                })
                .collect(),
        )
    }

    /// `collectNewGroup(doc)`.
    fn collect_new_group(&mut self, doc: i32, values: Vec<GroupSortValue>) {
        if !self.is_group_map_full() {
            let slot = self.group_map.len();
            let value = self.selector.copy_value();
            self.slots[slot] = values;
            let g = self.groups.len();
            self.groups.push(CollectedSearchGroup {
                group_value: value.clone(),
                top_doc: self.doc_base.saturating_add(doc),
                comparator_slot: slot,
            });
            self.group_map.insert(value, g);
            if self.is_group_map_full() {
                self.build_sorted_set();
                if let Some(last) = self.ordered.as_ref().and_then(TreeSet::last) {
                    self.bottom_slot = self.groups[last].comparator_slot;
                }
            }
            return;
        }
        let (sort, reversed) = (&self.sort, &self.reversed);
        let Some((ordered, bottom)) = self
            .ordered
            .as_mut()
            .and_then(|o| o.poll_last().map(|b| (o, b)))
        else {
            return;
        };
        self.group_map
            .remove(self.groups[bottom].group_value.as_ref());
        let value = self.selector.copy_value();
        self.groups[bottom].group_value = value.clone();
        self.groups[bottom].top_doc = self.doc_base.saturating_add(doc);
        let slot = self.groups[bottom].comparator_slot;
        self.slots[slot] = values;
        self.group_map.insert(value, bottom);
        let (slots, groups) = (&self.slots, &self.groups);
        ordered.add(bottom, &mut |a: &usize, b: &usize| {
            group_order(sort, reversed, slots, groups, *a, *b)
        });
        if let Some(last) = ordered.last() {
            self.bottom_slot = self.groups[last].comparator_slot;
        }
    }

    /// `collectExistingGroup(doc, group)`.
    fn collect_existing_group(&mut self, doc: i32, group: usize, values: Vec<GroupSortValue>) {
        if !beats(
            &self.sort,
            &self.reversed,
            &self.slots[self.groups[group].comparator_slot],
            &values,
        ) {
            return;
        }
        self.slots[self.spare_slot] = values;
        let (sort, reversed) = (&self.sort, &self.reversed);
        let mut skip_heavy_ops = false;
        let mut prev_last = None;
        if let Some(ordered) = self.ordered.as_mut() {
            prev_last = ordered.last();
            if ordered.first() == Some(group) {
                skip_heavy_ops = true;
            } else {
                let (slots, groups) = (&self.slots, &self.groups);
                ordered.remove(&group, &mut |a: &usize, b: &usize| {
                    group_order(sort, reversed, slots, groups, *a, *b)
                });
            }
        }
        self.groups[group].top_doc = self.doc_base.saturating_add(doc);
        std::mem::swap(
            &mut self.spare_slot,
            &mut self.groups[group].comparator_slot,
        );
        if let Some(ordered) = self.ordered.as_mut() {
            if !skip_heavy_ops {
                let (slots, groups) = (&self.slots, &self.groups);
                ordered.add(group, &mut |a: &usize, b: &usize| {
                    group_order(sort, reversed, slots, groups, *a, *b)
                });
            }
            let new_last = ordered.last();
            if Some(group) == new_last || prev_last != new_last {
                if let Some(last) = new_last {
                    self.bottom_slot = self.groups[last].comparator_slot;
                }
            }
        }
    }
}

/// `buildSortedSet`'s comparator: the slots' values, then the top document.
fn group_order<V>(
    sort: &Sort,
    reversed: &[i32],
    slots: &[Vec<GroupSortValue>],
    groups: &[CollectedSearchGroup<V>],
    a: usize,
    b: usize,
) -> Ordering {
    let (ga, gb) = (&groups[a], &groups[b]);
    match compare_all(
        sort,
        reversed,
        &slots[ga.comparator_slot],
        &slots[gb.comparator_slot],
    ) {
        Ordering::Equal => ga.top_doc.cmp(&gb.top_doc),
        c => c,
    }
}

impl<'a, S: GroupSelector<'a>> SegmentCollector<'a> for FirstPassGroupingCollector<'a, S> {
    fn score_mode(&self) -> ScoreMode {
        if self.needs_scores {
            ScoreMode::Complete
        } else {
            ScoreMode::CompleteNoScores
        }
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.doc_base = leaf.doc_base;
        self.keys = Some(LeafKeys::open(&self.sort, leaf)?);
        self.selector.set_next_reader(ord, leaf)?;
        // `setScorer`: `groupSelector.setScorer(scorer)`.
        self.selector.set_scorer()
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        let keys = self.keys.as_mut().ok_or_else(no_leaf)?;
        let mut values = None;
        // `isCompetitive(doc)`: once the top groups are full, a document
        // that cannot beat the bottom group is skipped before its group is
        // even looked at.
        if self.ordered.is_some() {
            let v = keys.values(&self.sort, doc, score)?;
            if !beats(
                &self.sort,
                &self.reversed,
                &self.slots[self.bottom_slot],
                &v,
            ) {
                return Ok(());
            }
            values = Some(v);
        }
        let state = self.selector.advance_to(doc, score)?;
        if self.ignore_docs_without_group_field && state == GroupState::Skip {
            return Ok(());
        }
        let group = self.group_map.get(self.selector.current_value());
        let values = match values {
            Some(v) => v,
            None => keys.values(&self.sort, doc, score)?,
        };
        match group {
            None => self.collect_new_group(doc, values),
            Some(g) => self.collect_existing_group(doc, g, values),
        }
        Ok(())
    }
}

/// `FirstPassGroupingCollectorManager<T>`: a first pass of `groupOffset +
/// topNGroups` groups per slice, merged by [`SearchGroup::merge`].
pub struct FirstPassGroupingCollectorManager<F> {
    selector_factory: F,
    group_sort: Sort,
    group_offset: usize,
    top_n_groups: usize,
    ignore_docs_without_group_field: bool,
}

impl<F> FirstPassGroupingCollectorManager<F> {
    /// `new FirstPassGroupingCollectorManager(groupSelectorFactory,
    /// groupSort, groupOffset, topNGroups, ignoreDocsWithoutGroupField)`.
    ///
    /// # Errors
    /// `topNGroups must be >= 1` ([`Error::IllegalArgument`]).
    pub fn new(
        selector_factory: F,
        group_sort: Sort,
        group_offset: usize,
        top_n_groups: usize,
        ignore_docs_without_group_field: bool,
    ) -> Result<Self> {
        if top_n_groups < 1 {
            return Err(Error::IllegalArgument(format!(
                "topNGroups must be >= 1 (got {top_n_groups})"
            )));
        }
        Ok(Self {
            selector_factory,
            group_sort,
            group_offset,
            top_n_groups,
            ignore_docs_without_group_field,
        })
    }
}

impl<'a, S, F> GroupingCollectorManager<'a> for FirstPassGroupingCollectorManager<F>
where
    S: GroupSelector<'a>,
    F: Fn() -> S,
{
    type Collector = FirstPassGroupingCollector<'a, S>;
    type Output = Vec<SearchGroup<S::Value>>;

    fn new_collector(&self) -> Result<Self::Collector> {
        FirstPassGroupingCollector::new(
            (self.selector_factory)(),
            self.group_sort.clone(),
            self.group_offset.saturating_add(self.top_n_groups),
            self.ignore_docs_without_group_field,
        )
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        let all: Vec<Vec<SearchGroup<S::Value>>> = collectors
            .into_iter()
            .filter_map(|mut c| c.top_groups(0))
            .collect();
        Ok(SearchGroup::merge(
            &all,
            self.group_offset,
            self.top_n_groups,
            &self.group_sort,
        ))
    }
}

// ---------------------------------------------------------------------------
// Second pass
// ---------------------------------------------------------------------------

/// `SecondPassGroupingCollector<T>`: hands every document of the given
/// groups to a [`GroupReducer`], counting all hits and the grouped ones.
pub struct SecondPassGroupingCollector<'a, S: GroupSelector<'a>, R> {
    selector: S,
    groups: Vec<SearchGroup<S::Value>>,
    reducer: R,
    total_hit_count: i32,
    total_grouped_hit_count: i32,
    /// Whether the search hands the collector a scorer (`setScorer`); not
    /// when a `CachingCollector` without scores replays into it.
    pub(crate) scorer: bool,
    _leaf: PhantomData<&'a ()>,
}

impl<'a, S: GroupSelector<'a>, R: GroupReducer<'a, S::Value>>
    SecondPassGroupingCollector<'a, S, R>
{
    /// `new SecondPassGroupingCollector(groupSelector, groups, reducer)`:
    /// the selector and the reducer restricted to `groups`.
    ///
    /// # Errors
    /// `no groups to collect (groups is empty)` ([`Error::IllegalArgument`]).
    pub fn new(
        mut selector: S,
        groups: Vec<SearchGroup<S::Value>>,
        mut reducer: R,
    ) -> Result<Self> {
        if groups.is_empty() {
            return Err(Error::IllegalArgument(
                "no groups to collect (groups is empty)".into(),
            ));
        }
        selector.set_groups(&groups);
        reducer.set_groups(&groups);
        Ok(Self {
            selector,
            groups,
            reducer,
            total_hit_count: 0,
            total_grouped_hit_count: 0,
            scorer: true,
            _leaf: PhantomData,
        })
    }

    /// `getGroupSelector()`.
    pub fn group_selector(&self) -> &S {
        &self.selector
    }

    /// The reducer.
    pub fn reducer(&self) -> &R {
        &self.reducer
    }

    /// The groups collected, in the order given.
    pub fn groups(&self) -> &[SearchGroup<S::Value>] {
        &self.groups
    }

    /// `totalHitCount`.
    pub fn total_hit_count(&self) -> i32 {
        self.total_hit_count
    }

    /// `totalGroupedHitCount`.
    pub fn total_grouped_hit_count(&self) -> i32 {
        self.total_grouped_hit_count
    }
}

impl<'a, S, R> SegmentCollector<'a> for SecondPassGroupingCollector<'a, S, R>
where
    S: GroupSelector<'a>,
    R: GroupReducer<'a, S::Value>,
{
    fn score_mode(&self) -> ScoreMode {
        if self.reducer.needs_scores() {
            ScoreMode::Complete
        } else {
            ScoreMode::CompleteNoScores
        }
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.reducer.set_next_reader(ord, leaf)?;
        self.selector.set_next_reader(ord, leaf)?;
        // `setScorer`: `groupSelector.setScorer(scorer)` -- which a cache
        // replayed without scores never calls.
        if self.scorer {
            self.selector.set_scorer()?;
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.total_hit_count = self.total_hit_count.wrapping_add(1);
        if self.selector.advance_to(doc, score)? == GroupState::Skip {
            return Ok(());
        }
        self.total_grouped_hit_count = self.total_grouped_hit_count.wrapping_add(1);
        self.reducer
            .collect(self.selector.current_value(), doc, score)
    }
}

/// `GroupReducer<T, C>` as Java defines it: a collector per group, made by
/// `newCollector()` (here `new_collector`), each entering every segment.
pub struct CollectorsReducer<V, C, F> {
    groups: GroupIndex<V>,
    collectors: Vec<C>,
    new_collector: F,
    needs_scores: bool,
}

impl<V: Eq + Hash, C, F: Fn() -> C> CollectorsReducer<V, C, F> {
    /// A reducer whose collectors `new_collector` makes; `needs_scores` is
    /// `needsScores()`.
    pub fn new(new_collector: F, needs_scores: bool) -> Self {
        Self {
            groups: GroupIndex::default(),
            collectors: Vec::new(),
            new_collector,
            needs_scores,
        }
    }

    /// `getCollector(value)`.
    pub fn collector(&self, value: Option<&V>) -> Option<&C> {
        self.groups.get(value).and_then(|i| self.collectors.get(i))
    }
}

impl<'a, V, C, F> GroupReducer<'a, V> for CollectorsReducer<V, C, F>
where
    V: Clone + Eq + Hash,
    C: SegmentCollector<'a>,
    F: Fn() -> C,
{
    fn set_groups(&mut self, groups: &[SearchGroup<V>]) {
        for g in groups {
            // `groups.put(value, new GroupCollector(newCollector()))`: a
            // repeated value replaces its collector.
            let c = (self.new_collector)();
            match self.groups.get(g.group_value.as_ref()) {
                Some(i) => self.collectors[i] = c,
                None => {
                    self.groups
                        .insert(g.group_value.clone(), self.collectors.len());
                    self.collectors.push(c);
                }
            }
        }
    }

    fn needs_scores(&self) -> bool {
        self.needs_scores
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        for c in &mut self.collectors {
            c.set_next_reader(ord, leaf)?;
        }
        Ok(())
    }

    fn collect(&mut self, value: Option<&V>, doc: i32, score: f32) -> Result<()> {
        let i = self.groups.get(value).ok_or_else(not_given)?;
        self.collectors[i].collect(doc, score)
    }
}

// ---------------------------------------------------------------------------
// TopGroupsCollector
// ---------------------------------------------------------------------------

/// The per-group collector of [`TopGroupsCollector`] (`TopDocsReducer`'s
/// `TopDocsAndMaxScoreCollector`): a `TopScoreDocCollector` when the
/// within-group sort is `Sort.RELEVANCE` itself, else a
/// `TopFieldCollector` with, when max scores are asked for, a
/// `MaxScoreCollector`. Counts every hit (`totalHitsThreshold =
/// Integer.MAX_VALUE`).
pub struct GroupTopDocs<'a> {
    by_score: bool,
    sort: Sort,
    reversed: Vec<i32>,
    num_hits: usize,
    /// The top hits, best first.
    hits: Vec<GroupScoreDoc>,
    total_hits: u64,
    /// `MaxScoreCollector`: `None` when not asked for; else the maximum
    /// (from `Float.MIN_VALUE`) and whether anything was collected.
    max_score: Option<(f32, bool)>,
    keys: Option<LeafKeys<'a>>,
    doc_base: i32,
}

impl<'a> GroupTopDocs<'a> {
    /// `TopScoreDocCollectorManager(numHits, null, Integer.MAX_VALUE)` when
    /// `by_score`, else `TopFieldCollectorManager(sort, numHits, null,
    /// Integer.MAX_VALUE)` (with a `MaxScoreCollector` when
    /// `track_max_score`).
    ///
    /// # Errors
    /// `numHits must be > 0` ([`Error::IllegalArgument`]).
    pub fn new(
        by_score: bool,
        sort: &Sort,
        num_hits: usize,
        track_max_score: bool,
    ) -> Result<Self> {
        if num_hits == 0 {
            return Err(Error::IllegalArgument(if by_score {
                "numHits must be > 0; please use TotalHitCountCollectorManager if you just need \
                 the total hit count"
                    .into()
            } else {
                "numHits must be > 0; please use TotalHitCountCollector if you just need the \
                 total hit count"
                    .into()
            }));
        }
        Ok(Self::build(by_score, sort, num_hits, track_max_score))
    }

    /// [`Self::new`] past its check.
    fn build(by_score: bool, sort: &Sort, num_hits: usize, track_max_score: bool) -> Self {
        Self {
            by_score,
            reversed: sort.reversed(),
            sort: sort.clone(),
            num_hits,
            hits: Vec::new(),
            total_hits: 0,
            max_score: (!by_score && track_max_score).then_some((FLOAT_MIN_VALUE, false)),
            keys: None,
            doc_base: 0,
        }
    }

    /// The hits from `start`, at most `how_many` (`topDocs(start,
    /// howMany)`).
    pub fn top_docs(&self, start: usize, how_many: usize) -> Vec<GroupScoreDoc> {
        if start >= self.hits.len() || how_many == 0 {
            return Vec::new();
        }
        let end = start.saturating_add(how_many).min(self.hits.len());
        self.hits[start..end].to_vec()
    }

    /// Every hit kept, best first (`topDocs()`).
    pub fn all(&self) -> &[GroupScoreDoc] {
        &self.hits
    }

    /// `totalHits`.
    pub fn total_hits(&self) -> TotalHits {
        TotalHits {
            value: self.total_hits,
            relation: TotalHitsRelation::EqualTo,
        }
    }

    /// `MaxScoreCollector.getMaxScore()`: `NaN` before any hit; `None`
    /// without one.
    pub fn max_score(&self) -> Option<f32> {
        self.max_score
            .map(|(m, any)| if any { m } else { f32::NAN })
    }

    /// Whether the hits are by score (`sortedByScore`).
    pub fn sorted_by_score(&self) -> bool {
        self.by_score
    }
}

impl<'a> SegmentCollector<'a> for GroupTopDocs<'a> {
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::Complete
    }

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.doc_base = leaf.doc_base;
        if !self.by_score {
            self.keys = Some(LeafKeys::open(&self.sort, leaf)?);
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.total_hits = self.total_hits.saturating_add(1);
        if let Some((max, any)) = self.max_score.as_mut() {
            *any = true;
            *max = java_max_f32(score, *max);
        }
        let global = self.doc_base.saturating_add(doc);
        if self.by_score {
            // `TopScoreDocCollector`: a full queue takes only a strictly
            // higher score; documents arrive in order, so ties keep the
            // earlier one.
            if self.hits.len() >= self.num_hits {
                let worst = self.hits.last().map_or(f32::NEG_INFINITY, |h| h.score);
                if score <= worst || score.is_nan() {
                    return Ok(());
                }
            }
            let at = self.hits.partition_point(|h| h.score >= score);
            self.hits.insert(
                at,
                GroupScoreDoc {
                    doc: global,
                    score,
                    fields: None,
                    shard_index: -1,
                },
            );
        } else {
            let keys = self.keys.as_mut().ok_or_else(no_leaf)?;
            let values = keys.values(&self.sort, doc, score)?;
            // `TopFieldCollector`: a full queue takes only a document
            // strictly better than its bottom.
            if self.hits.len() >= self.num_hits {
                let bottom = self.hits.last().and_then(|h| h.fields.as_deref());
                if !bottom.is_some_and(|b| beats(&self.sort, &self.reversed, b, &values)) {
                    return Ok(());
                }
            }
            let (sort, reversed) = (&self.sort, &self.reversed);
            let at = self.hits.partition_point(|h| {
                h.fields
                    .as_deref()
                    .is_some_and(|f| compare_all(sort, reversed, f, &values) != Ordering::Greater)
            });
            self.hits.insert(
                at,
                GroupScoreDoc {
                    doc: global,
                    score: f32::NAN,
                    fields: Some(values),
                    shard_index: -1,
                },
            );
        }
        self.hits.truncate(self.num_hits);
        Ok(())
    }
}

/// The reducer of [`TopGroupsCollector`] (`TopDocsReducer`).
pub type TopDocsReducer<'a, V> =
    CollectorsReducer<V, GroupTopDocs<'a>, Box<dyn Fn() -> GroupTopDocs<'a> + 'a>>;

/// `TopGroupsCollector<T>`: the second pass collecting each top group's
/// top documents by `withinGroupSort`.
pub struct TopGroupsCollector<'a, S: GroupSelector<'a>> {
    inner: SecondPassGroupingCollector<'a, S, TopDocsReducer<'a, S::Value>>,
    group_sort: Sort,
    within_group_sort: Sort,
    max_docs_per_group: usize,
}

impl<'a, S: GroupSelector<'a>> TopGroupsCollector<'a, S> {
    /// `new TopGroupsCollector(groupSelector, groups, groupSort,
    /// withinGroupSort, maxDocsPerGroup, getMaxScores)`.
    ///
    /// # Errors
    /// No groups, or `maxDocsPerGroup` of `0` (`numHits must be > 0`), as
    /// [`Error::IllegalArgument`].
    pub fn new(
        selector: S,
        groups: Vec<SearchGroup<S::Value>>,
        group_sort: Sort,
        within_group_sort: Sort,
        max_docs_per_group: usize,
        get_max_scores: bool,
    ) -> Result<Self> {
        let by_score = within_group_sort.is_relevance_singleton();
        // The collector constructor's own check, which Java reaches through
        // `setGroups` building the first group's collector.
        GroupTopDocs::new(
            by_score,
            &within_group_sort,
            max_docs_per_group,
            get_max_scores,
        )?;
        let needs_scores = get_max_scores || within_group_sort.needs_scores();
        let sort = within_group_sort.clone();
        let factory: Box<dyn Fn() -> GroupTopDocs<'a> + 'a> = Box::new(move || {
            GroupTopDocs::build(by_score, &sort, max_docs_per_group, get_max_scores)
        });
        let reducer = CollectorsReducer::new(factory, needs_scores);
        Ok(Self {
            inner: SecondPassGroupingCollector::new(selector, groups, reducer)?,
            group_sort,
            within_group_sort,
            max_docs_per_group,
        })
    }

    /// The second pass underneath.
    pub fn second_pass(&self) -> &SecondPassGroupingCollector<'a, S, TopDocsReducer<'a, S::Value>> {
        &self.inner
    }

    /// See [`SecondPassGroupingCollector::scorer`].
    pub(crate) fn set_has_scorer(&mut self, scorer: bool) {
        self.inner.scorer = scorer;
    }

    /// `getTopGroups(withinGroupOffset)`: every group, in the order given,
    /// with its documents from `within_group_offset`.
    pub fn top_groups(&self, within_group_offset: usize) -> TopGroups<S::Value> {
        let mut groups = Vec::with_capacity(self.inner.groups.len());
        let mut max_score = FLOAT_MIN_VALUE;
        for group in &self.inner.groups {
            // Every group has its collector (`setGroups`).
            let Some(c) = self.inner.reducer.collector(group.group_value.as_ref()) else {
                continue;
            };
            let (score_docs, group_max_score) = if c.sorted_by_score() {
                let all = c.all();
                let group_max = all.first().map_or(f32::NAN, |h| h.score);
                let docs = if all.len() <= within_group_offset {
                    Vec::new()
                } else {
                    let end = within_group_offset
                        .saturating_add(self.max_docs_per_group)
                        .min(all.len());
                    all[within_group_offset..end].to_vec()
                };
                (docs, group_max)
            } else {
                (
                    c.top_docs(within_group_offset, self.max_docs_per_group),
                    c.max_score().unwrap_or(f32::NAN),
                )
            };
            let total_hits = c.total_hits();
            max_score = non_nan_max(max_score, group_max_score);
            groups.push(GroupDocs {
                score: f32::NAN,
                max_score: group_max_score,
                total_hits,
                score_docs,
                group_value: group.group_value.clone(),
                group_sort_values: group.sort_values.clone(),
            });
        }
        TopGroups {
            total_hit_count: self.inner.total_hit_count,
            total_grouped_hit_count: self.inner.total_grouped_hit_count,
            total_group_count: None,
            groups,
            group_sort: self.group_sort.fields.clone(),
            within_group_sort: self.within_group_sort.fields.clone(),
            max_score,
        }
    }
}

impl<'a, S: GroupSelector<'a>> SegmentCollector<'a> for TopGroupsCollector<'a, S> {
    fn score_mode(&self) -> ScoreMode {
        self.inner.score_mode()
    }
    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.inner.set_next_reader(ord, leaf)
    }
    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.inner.collect(doc, score)
    }
}

/// `TopGroupsCollectorManager<T>`: a [`TopGroupsCollector`] per slice over
/// the same groups (each keeping `withinGroupOffset + maxDocsPerGroup`
/// documents), merged by [`TopGroups::merge`].
pub struct TopGroupsCollectorManager<F, V> {
    selector_factory: F,
    search_groups: Vec<SearchGroup<V>>,
    group_sort: Sort,
    sort_within_group: Sort,
    within_group_offset: usize,
    max_docs_per_group: usize,
    get_max_scores: bool,
    score_merge_mode: ScoreMergeMode,
}

impl<F, V> TopGroupsCollectorManager<F, V> {
    /// `new TopGroupsCollectorManager(groupSelectorFactory, searchGroups,
    /// groupSort, sortWithinGroup, withinGroupOffset, maxDocsPerGroup,
    /// getMaxScores, scoreMergeMode)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        selector_factory: F,
        search_groups: Vec<SearchGroup<V>>,
        group_sort: Sort,
        sort_within_group: Sort,
        within_group_offset: usize,
        max_docs_per_group: usize,
        get_max_scores: bool,
        score_merge_mode: ScoreMergeMode,
    ) -> Self {
        Self {
            selector_factory,
            search_groups,
            group_sort,
            sort_within_group,
            within_group_offset,
            max_docs_per_group,
            get_max_scores,
            score_merge_mode,
        }
    }
}

impl<'a, S, F> GroupingCollectorManager<'a> for TopGroupsCollectorManager<F, S::Value>
where
    S: GroupSelector<'a>,
    F: Fn() -> S,
{
    type Collector = TopGroupsCollector<'a, S>;
    /// `null` (`None`) only for no collectors.
    type Output = Option<TopGroups<S::Value>>;

    fn new_collector(&self) -> Result<Self::Collector> {
        TopGroupsCollector::new(
            (self.selector_factory)(),
            self.search_groups.clone(),
            self.group_sort.clone(),
            self.sort_within_group.clone(),
            self.within_group_offset
                .saturating_add(self.max_docs_per_group),
            self.get_max_scores,
        )
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        let shards: Vec<TopGroups<S::Value>> = collectors.iter().map(|c| c.top_groups(0)).collect();
        TopGroups::merge(
            &shards,
            &self.group_sort,
            &self.sort_within_group,
            self.within_group_offset,
            self.max_docs_per_group,
            self.score_merge_mode,
        )
    }
}

// ---------------------------------------------------------------------------
// AllGroupsCollector
// ---------------------------------------------------------------------------

/// `AllGroupsCollector<T>`: every group a matching document is in (the
/// documents without a value as the `None` group).
pub struct AllGroupsCollector<'a, S: GroupSelector<'a>> {
    selector: S,
    index: GroupIndex<S::Value>,
    groups: Vec<Option<S::Value>>,
}

impl<'a, S: GroupSelector<'a>> AllGroupsCollector<'a, S> {
    /// `new AllGroupsCollector(groupSelector)`.
    pub fn new(selector: S) -> Self {
        Self {
            selector,
            index: GroupIndex::default(),
            groups: Vec::new(),
        }
    }

    /// The selector (`GroupingSearch` shares one with the first pass).
    pub(crate) fn selector_mut(&mut self) -> &mut S {
        &mut self.selector
    }

    /// `getGroupCount()`.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// `getGroups()`, in first-seen order.
    pub fn groups(&self) -> &[Option<S::Value>] {
        &self.groups
    }
}

impl<'a, S: GroupSelector<'a>> SegmentCollector<'a> for AllGroupsCollector<'a, S> {
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.selector.set_next_reader(ord, leaf)
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.selector.advance_to(doc, score)?;
        if self.index.get(self.selector.current_value()).is_some() {
            return Ok(());
        }
        let v = self.selector.copy_value();
        self.index.insert(v.clone(), self.groups.len());
        self.groups.push(v);
        Ok(())
    }
}

/// `AllGroupsCollectorManager<T>`: the union of the slices' groups.
pub struct AllGroupsCollectorManager<F> {
    selector_factory: F,
}

impl<F> AllGroupsCollectorManager<F> {
    /// `new AllGroupsCollectorManager(groupSelectorFactory)`.
    pub fn new(selector_factory: F) -> Self {
        Self { selector_factory }
    }
}

impl<'a, S, F> GroupingCollectorManager<'a> for AllGroupsCollectorManager<F>
where
    S: GroupSelector<'a>,
    F: Fn() -> S,
{
    type Collector = AllGroupsCollector<'a, S>;
    type Output = Vec<Option<S::Value>>;

    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(AllGroupsCollector::new((self.selector_factory)()))
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        let mut index = GroupIndex::default();
        let mut out = Vec::new();
        for c in collectors {
            for g in c.groups {
                if index.get(g.as_ref()).is_none() {
                    index.insert(g.clone(), out.len());
                    out.push(g);
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// AllGroupHeadsCollector
// ---------------------------------------------------------------------------

/// `AllGroupHeadsCollector.GroupHead<T>`: a group's best document so far
/// (global id) and its sort values.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupHead<V> {
    pub group_value: Option<V>,
    pub doc: i32,
    pub sort_values: Vec<GroupSortValue>,
}

/// `AllGroupHeadsCollector<T>`: the most relevant document of every group
/// (`ScoringGroupHeadsCollector` when the sort is `Sort.RELEVANCE`,
/// `SortingGroupHeadsCollector` otherwise).
pub struct AllGroupHeadsCollector<'a, S: GroupSelector<'a>> {
    selector: S,
    sort: Sort,
    reversed: Vec<i32>,
    scoring: bool,
    index: GroupIndex<S::Value>,
    heads: Vec<GroupHead<S::Value>>,
    keys: Option<LeafKeys<'a>>,
    doc_base: i32,
}

impl<'a, S: GroupSelector<'a>> AllGroupHeadsCollector<'a, S> {
    /// `AllGroupHeadsCollector.newCollector(selector, sort)`.
    pub fn new(selector: S, sort: Sort) -> Self {
        Self {
            selector,
            reversed: sort.reversed(),
            scoring: sort.is_relevance(),
            sort,
            index: GroupIndex::default(),
            heads: Vec::new(),
            keys: None,
            doc_base: 0,
        }
    }

    /// The selector (`GroupingSearch` shares one with the first pass).
    pub(crate) fn selector_mut(&mut self) -> &mut S {
        &mut self.selector
    }

    /// `getCollectedGroupHeads()`, in first-seen order.
    pub fn group_heads(&self) -> &[GroupHead<S::Value>] {
        &self.heads
    }

    /// `retrieveGroupHeads()`: the heads' documents.
    pub fn retrieve_group_heads(&self) -> Vec<i32> {
        self.heads.iter().map(|h| h.doc).collect()
    }

    /// `retrieveGroupHeads(maxDoc)`: the heads as a bit set.
    pub fn retrieve_group_heads_bits(&self, max_doc: usize) -> FixedBitSet {
        heads_bits(self.heads.iter().map(|h| h.doc), max_doc)
    }

    /// `groupHeadsSize()`.
    pub fn group_heads_size(&self) -> usize {
        self.heads.len()
    }
}

fn heads_bits(docs: impl Iterator<Item = i32>, max_doc: usize) -> FixedBitSet {
    let mut bits = FixedBitSet::new(max_doc);
    for doc in docs {
        if let Ok(d) = usize::try_from(doc) {
            // FBS: a head is a document below the reader's `maxDoc`, the
            // set's length; the check keeps a wrong `max_doc` from
            // panicking (Java would throw).
            if d < bits.len() {
                bits.set(d);
            }
        }
    }
    bits
}

impl<'a, S: GroupSelector<'a>> SegmentCollector<'a> for AllGroupHeadsCollector<'a, S> {
    fn score_mode(&self) -> ScoreMode {
        if self.sort.needs_scores() {
            ScoreMode::Complete
        } else {
            ScoreMode::CompleteNoScores
        }
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.selector.set_next_reader(ord, leaf)?;
        self.doc_base = leaf.doc_base;
        if !self.scoring {
            self.keys = Some(LeafKeys::open(&self.sort, leaf)?);
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.selector.advance_to(doc, score)?;
        let global = self.doc_base.saturating_add(doc);
        let Some(h) = self.index.get(self.selector.current_value()) else {
            let sort_values = if self.scoring {
                vec![GroupSortValue::Float(score)]
            } else {
                let keys = self.keys.as_mut().ok_or_else(no_leaf)?;
                keys.values(&self.sort, doc, score)?
            };
            let v = self.selector.copy_value();
            self.index.insert(v.clone(), self.heads.len());
            self.heads.push(GroupHead {
                group_value: v,
                doc: global,
                sort_values,
            });
            return Ok(());
        };
        let head = &mut self.heads[h];
        if self.scoring {
            // `ScoringGroupHead.compare`: `Float.compare(score, topScore)`.
            let top = float_of(&head.sort_values);
            if super::sort::float_compare(score, top) == Ordering::Greater {
                head.doc = global;
                head.sort_values = vec![GroupSortValue::Float(score)];
            }
            return Ok(());
        }
        let keys = self.keys.as_mut().ok_or_else(no_leaf)?;
        let values = keys.values(&self.sort, doc, score)?;
        if beats(&self.sort, &self.reversed, &head.sort_values, &values) {
            head.doc = global;
            head.sort_values = values;
        }
        Ok(())
    }
}

/// `AllGroupHeadsCollectorManager.GroupHeadsResult`: the heads' documents.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupHeadsResult {
    pub(crate) group_heads: Vec<i32>,
}

impl GroupHeadsResult {
    /// `retrieveGroupHeads()`.
    pub fn retrieve_group_heads(&self) -> &[i32] {
        &self.group_heads
    }

    /// `retrieveGroupHeads(maxDoc)`.
    pub fn retrieve_group_heads_bits(&self, max_doc: usize) -> FixedBitSet {
        heads_bits(self.group_heads.iter().copied(), max_doc)
    }
}

/// `AllGroupHeadsCollectorManager<T>`: each slice's heads, merged group by
/// group to the most relevant (ties to the lower document).
pub struct AllGroupHeadsCollectorManager<F> {
    selector_factory: F,
    sort_within_group: Sort,
}

impl<F> AllGroupHeadsCollectorManager<F> {
    /// `new AllGroupHeadsCollectorManager(groupSelectorFactory,
    /// sortWithinGroup)`.
    pub fn new(selector_factory: F, sort_within_group: Sort) -> Self {
        Self {
            selector_factory,
            sort_within_group,
        }
    }
}

impl<'a, S, F> GroupingCollectorManager<'a> for AllGroupHeadsCollectorManager<F>
where
    S: GroupSelector<'a>,
    F: Fn() -> S,
{
    type Collector = AllGroupHeadsCollector<'a, S>;
    type Output = GroupHeadsResult;

    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(AllGroupHeadsCollector::new(
            (self.selector_factory)(),
            self.sort_within_group.clone(),
        ))
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        let relevance = self.sort_within_group.is_relevance();
        let reversed = self.sort_within_group.reversed();
        let mut index: GroupIndex<S::Value> = GroupIndex::default();
        let mut merged: Vec<(i32, Vec<GroupSortValue>)> = Vec::new();
        for c in collectors {
            for head in c.heads {
                let Some(i) = index.get(head.group_value.as_ref()) else {
                    index.insert(head.group_value, merged.len());
                    merged.push((head.doc, head.sort_values));
                    continue;
                };
                let (doc, values) = &merged[i];
                let competitive = if relevance {
                    let (a, b) = (float_of(&head.sort_values), float_of(values));
                    match super::sort::float_compare(a, b) {
                        Ordering::Greater => true,
                        Ordering::Equal => head.doc < *doc,
                        Ordering::Less => false,
                    }
                } else {
                    let mut cmp = Ordering::Equal;
                    for (k, f) in self.sort_within_group.fields.iter().enumerate() {
                        let c = compare_values(f, &head.sort_values[k], &values[k]);
                        let c = if reversed[k] < 0 { c.reverse() } else { c };
                        if c != Ordering::Equal {
                            cmp = c;
                            break;
                        }
                    }
                    cmp == Ordering::Less || (cmp == Ordering::Equal && head.doc < *doc)
                };
                if competitive {
                    merged[i] = (head.doc, head.sort_values);
                }
            }
        }
        Ok(GroupHeadsResult {
            group_heads: merged.into_iter().map(|(d, _)| d).collect(),
        })
    }
}

fn float_of(values: &[GroupSortValue]) -> f32 {
    match values.first() {
        Some(GroupSortValue::Float(f)) => *f,
        _ => f32::NAN,
    }
}

// ---------------------------------------------------------------------------
// DistinctValuesCollector
// ---------------------------------------------------------------------------

/// `DistinctValuesCollector.GroupCount<T, R>`: a group and the distinct
/// values of its documents (`None` for a document without one), in
/// first-seen order.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupCount<V, R> {
    pub group_value: Option<V>,
    pub unique_values: Vec<Option<R>>,
}

/// One group's distinct values: an index over them and the values in
/// first-seen order.
type DistinctSet<R> = (GroupIndex<R>, Vec<Option<R>>);

/// The reducer of [`DistinctValuesCollector`] (`DistinctValuesReducer`):
/// the value selector, once, and each group's distinct values. (Java gives
/// every group's `ValuesCollector` the same selector; here the reducer owns
/// it and moves it once per document.)
pub struct DistinctValuesReducer<'a, V, R: GroupSelector<'a>> {
    value_selector: R,
    groups: GroupIndex<V>,
    values: Vec<DistinctSet<R::Value>>,
}

impl<'a, V: Eq + Hash, R: GroupSelector<'a>> DistinctValuesReducer<'a, V, R> {
    fn new(value_selector: R) -> Self {
        Self {
            value_selector,
            groups: GroupIndex::default(),
            values: Vec::new(),
        }
    }
}

impl<'a, V: Clone + Eq + Hash, R: GroupSelector<'a>> GroupReducer<'a, V>
    for DistinctValuesReducer<'a, V, R>
{
    fn set_groups(&mut self, groups: &[SearchGroup<V>]) {
        for g in groups {
            let fresh = (GroupIndex::default(), Vec::new());
            match self.groups.get(g.group_value.as_ref()) {
                Some(i) => self.values[i] = fresh,
                None => {
                    self.groups.insert(g.group_value.clone(), self.values.len());
                    self.values.push(fresh);
                }
            }
        }
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        if self.values.is_empty() {
            return Ok(());
        }
        self.value_selector.set_next_reader(ord, leaf)
    }

    fn collect(&mut self, value: Option<&V>, doc: i32, score: f32) -> Result<()> {
        let i = self.groups.get(value).ok_or_else(not_given)?;
        // `ValuesCollector.collect`: the value if the selector accepts the
        // document, else `null`.
        let accepted = self.value_selector.advance_to(doc, score)? == GroupState::Accept;
        let v = if accepted {
            self.value_selector.current_value()
        } else {
            None
        };
        let (index, list) = &mut self.values[i];
        if index.get(v).is_none() {
            let owned = if accepted {
                self.value_selector.copy_value()
            } else {
                None
            };
            index.insert(owned.clone(), list.len());
            list.push(owned);
        }
        Ok(())
    }
}

/// `DistinctValuesCollector<T, R>`: the second pass collecting, per top
/// group, the distinct values `value_selector` gives its documents.
pub struct DistinctValuesCollector<'a, S: GroupSelector<'a>, R: GroupSelector<'a>> {
    inner: SecondPassGroupingCollector<'a, S, DistinctValuesReducer<'a, S::Value, R>>,
}

impl<'a, S: GroupSelector<'a>, R: GroupSelector<'a>> DistinctValuesCollector<'a, S, R> {
    /// `new DistinctValuesCollector(groupSelector, groups, valueSelector)`.
    ///
    /// # Errors
    /// No groups ([`Error::IllegalArgument`]).
    pub fn new(selector: S, groups: Vec<SearchGroup<S::Value>>, value_selector: R) -> Result<Self> {
        Ok(Self {
            inner: SecondPassGroupingCollector::new(
                selector,
                groups,
                DistinctValuesReducer::new(value_selector),
            )?,
        })
    }

    /// `getGroups()`: each group, in the order given, with its values.
    pub fn groups(&self) -> Vec<GroupCount<S::Value, R::Value>> {
        let r = &self.inner.reducer;
        self.inner
            .groups
            .iter()
            .map(|g| GroupCount {
                group_value: g.group_value.clone(),
                unique_values: r
                    .groups
                    .get(g.group_value.as_ref())
                    .map(|i| r.values[i].1.clone())
                    .unwrap_or_default(),
            })
            .collect()
    }

    /// The second pass underneath.
    pub fn second_pass(
        &self,
    ) -> &SecondPassGroupingCollector<'a, S, DistinctValuesReducer<'a, S::Value, R>> {
        &self.inner
    }
}

impl<'a, S: GroupSelector<'a>, R: GroupSelector<'a>> SegmentCollector<'a>
    for DistinctValuesCollector<'a, S, R>
{
    fn score_mode(&self) -> ScoreMode {
        self.inner.score_mode()
    }
    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.inner.set_next_reader(ord, leaf)
    }
    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.inner.collect(doc, score)
    }
}

/// `DistinctValuesCollectorManager<T, R>`: each slice's distinct values,
/// united group by group.
pub struct DistinctValuesCollectorManager<F, G, V> {
    selector_factory: F,
    search_groups: Vec<SearchGroup<V>>,
    value_selector_factory: G,
}

impl<F, G, V> DistinctValuesCollectorManager<F, G, V> {
    /// `new DistinctValuesCollectorManager(groupSelectorFactory,
    /// searchGroups, valueSelectorFactory)`.
    pub fn new(
        selector_factory: F,
        search_groups: Vec<SearchGroup<V>>,
        value_selector_factory: G,
    ) -> Self {
        Self {
            selector_factory,
            search_groups,
            value_selector_factory,
        }
    }
}

impl<'a, S, R, F, G> GroupingCollectorManager<'a> for DistinctValuesCollectorManager<F, G, S::Value>
where
    S: GroupSelector<'a>,
    R: GroupSelector<'a>,
    F: Fn() -> S,
    G: Fn() -> R,
{
    type Collector = DistinctValuesCollector<'a, S, R>;
    type Output = Vec<GroupCount<S::Value, R::Value>>;

    fn new_collector(&self) -> Result<Self::Collector> {
        DistinctValuesCollector::new(
            (self.selector_factory)(),
            self.search_groups.clone(),
            (self.value_selector_factory)(),
        )
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        let all: Vec<Vec<GroupCount<S::Value, R::Value>>> = collectors
            .iter()
            .map(DistinctValuesCollector::groups)
            .collect();
        let Some(first) = all.first() else {
            return Ok(Vec::new());
        };
        let mut merged = Vec::with_capacity(first.len());
        for (j, g) in first.iter().enumerate() {
            let mut index = GroupIndex::default();
            let mut union = Vec::new();
            for counts in &all {
                if let Some(c) = counts.get(j) {
                    for v in &c.unique_values {
                        if index.get(v.as_ref()).is_none() {
                            index.insert(v.clone(), union.len());
                            union.push(v.clone());
                        }
                    }
                }
            }
            merged.push(GroupCount {
                group_value: g.group_value.clone(),
                unique_values: union,
            });
        }
        Ok(merged)
    }
}
