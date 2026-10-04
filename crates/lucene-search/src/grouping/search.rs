//! `GroupingSearch`: the convenience that runs grouping end to end -- by a
//! [`GroupSelector`] in two passes (with the group count and the group
//! heads alongside the first, and the first pass optionally cached for the
//! second), or by document blocks in one.

use lucene_util::fixed_bit_set::FixedBitSet;

use super::block::BlockGroupingCollectorManager;
use super::collectors::{
    AllGroupHeadsCollector, AllGroupsCollector, FirstPassGroupingCollector, TopGroupsCollector,
};
use super::sort::Sort;
use super::{search_manager, GroupSelector, TopGroups};
use crate::collector::{ScoreMode, ScoringCollector};
use crate::collectors::{CachingCollector, NoOpCollector};
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::{search_segments, PerSegment, SegmentCollector};
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::{Error, Result};

/// What a [`GroupingSearch`] groups by.
enum Grouper<F> {
    /// `groupField`/`groupSelector`: a selector per collector, from the
    /// factory.
    Selector(F),
    /// `groupEndDocs`: blocks closed by the query's documents.
    Blocks(BooleanQuery),
}

/// What [`GroupingSearch::search`] found: the top groups, and what
/// `getAllMatchingGroups()`/`getAllGroupHeads()` return after it.
#[derive(Debug, Clone)]
pub struct GroupingSearchResult<V> {
    pub top_groups: TopGroups<V>,
    /// Every group a hit is in, when `setAllGroups(true)`; else empty.
    pub matching_groups: Vec<Option<V>>,
    /// Each group's head document, when `setAllGroupHeads(true)`; else
    /// none (`Bits.MatchNoBits`).
    pub matching_group_heads: FixedBitSet,
}

/// `GroupingSearch`, with its defaults: groups and documents within them by
/// relevance, one document per group, max scores on, no caching.
pub struct GroupingSearch<F> {
    grouper: Grouper<F>,
    group_sort: Sort,
    sort_within_group: Sort,
    group_docs_offset: usize,
    group_docs_limit: usize,
    include_max_score: bool,
    max_cache_ram_mb: Option<f64>,
    max_docs_to_cache: Option<usize>,
    cache_scores: bool,
    all_groups: bool,
    all_group_heads: bool,
    ignore_docs_without_group_field: bool,
}

impl GroupingSearch<()> {
    /// `new GroupingSearch(groupEndDocs)`: blocks of documents, each closed
    /// by a document `group_end_docs` matches ([`Self::search_blocks`]).
    pub fn by_blocks(group_end_docs: BooleanQuery) -> Self {
        Self::with(Grouper::Blocks(group_end_docs))
    }
}

impl<F> GroupingSearch<F> {
    fn with(grouper: Grouper<F>) -> Self {
        Self {
            grouper,
            group_sort: Sort::relevance(),
            sort_within_group: Sort::relevance(),
            group_docs_offset: 0,
            group_docs_limit: 1,
            include_max_score: true,
            max_cache_ram_mb: None,
            max_docs_to_cache: None,
            cache_scores: false,
            all_groups: false,
            all_group_heads: false,
            ignore_docs_without_group_field: false,
        }
    }

    /// `new GroupingSearch(groupSelector)` (and `new
    /// GroupingSearch(groupField)` with a
    /// [`TermGroupSelector`](super::TermGroupSelector) factory): every
    /// collector gets its own selector from `selector_factory`.
    pub fn new(selector_factory: F) -> Self {
        Self::with(Grouper::Selector(selector_factory))
    }

    /// `setCachingInMB(maxCacheRAMMB, cacheScores)`.
    pub fn set_caching_in_mb(mut self, max_cache_ram_mb: f64, cache_scores: bool) -> Self {
        self.max_cache_ram_mb = Some(max_cache_ram_mb);
        self.max_docs_to_cache = None;
        self.cache_scores = cache_scores;
        self
    }

    /// `setCaching(maxDocsToCache, cacheScores)`.
    pub fn set_caching(mut self, max_docs_to_cache: usize, cache_scores: bool) -> Self {
        self.max_docs_to_cache = Some(max_docs_to_cache);
        self.max_cache_ram_mb = None;
        self.cache_scores = cache_scores;
        self
    }

    /// `disableCaching()`.
    pub fn disable_caching(mut self) -> Self {
        self.max_cache_ram_mb = None;
        self.max_docs_to_cache = None;
        self
    }

    /// `setGroupSort(groupSort)`.
    pub fn set_group_sort(mut self, group_sort: Sort) -> Self {
        self.group_sort = group_sort;
        self
    }

    /// `setSortWithinGroup(sortWithinGroup)`.
    pub fn set_sort_within_group(mut self, sort_within_group: Sort) -> Self {
        self.sort_within_group = sort_within_group;
        self
    }

    /// `setGroupDocsOffset(groupDocsOffset)`.
    pub fn set_group_docs_offset(mut self, group_docs_offset: usize) -> Self {
        self.group_docs_offset = group_docs_offset;
        self
    }

    /// `setGroupDocsLimit(groupDocsLimit)`.
    pub fn set_group_docs_limit(mut self, group_docs_limit: usize) -> Self {
        self.group_docs_limit = group_docs_limit;
        self
    }

    /// `setIncludeMaxScore(includeMaxScore)`.
    pub fn set_include_max_score(mut self, include_max_score: bool) -> Self {
        self.include_max_score = include_max_score;
        self
    }

    /// `setAllGroups(allGroups)`.
    pub fn set_all_groups(mut self, all_groups: bool) -> Self {
        self.all_groups = all_groups;
        self
    }

    /// `setAllGroupHeads(allGroupHeads)`.
    pub fn set_all_group_heads(mut self, all_group_heads: bool) -> Self {
        self.all_group_heads = all_group_heads;
        self
    }

    /// `setIgnoreDocsWithoutGroupField(ignoreDocsWithoutGroupField)`.
    pub fn set_ignore_docs_without_group_field(mut self, ignore: bool) -> Self {
        self.ignore_docs_without_group_field = ignore;
        self
    }

    /// `search(searcher, query, groupOffset, groupLimit)` for grouping by
    /// blocks (`groupByDocBlock`): a [`BlockGroupingCollectorManager`] run
    /// per slice.
    ///
    /// # Errors
    /// [`Error::IllegalState`] for a search made with a selector; Java's
    /// argument errors; what the search reports.
    pub fn search_blocks(
        &self,
        searcher: &IndexSearcher<'_, '_>,
        query: &BooleanQuery,
        group_offset: usize,
        group_limit: usize,
    ) -> Result<TopGroups<()>> {
        let Grouper::Blocks(end_docs) = &self.grouper else {
            return Err(Error::IllegalState(
                "Either groupField, groupFunction or groupEndDocs must be set.".into(),
            ));
        };
        let manager = BlockGroupingCollectorManager::new(
            self.group_sort.clone(),
            group_offset,
            group_limit,
            self.group_sort.needs_scores() || self.sort_within_group.needs_scores(),
            end_docs.clone(),
            self.sort_within_group.clone(),
            self.group_docs_offset,
            self.group_docs_offset.saturating_add(self.group_docs_limit),
        )?;
        search_manager(searcher, query, &manager)
    }

    /// `search(searcher, query, groupOffset, groupLimit)` by a selector
    /// (`groupByFieldOrFunction`): the first pass (with
    /// [`AllGroupsCollector`] and [`AllGroupHeadsCollector`] when asked
    /// for, and cached when caching is on), then [`TopGroupsCollector`]
    /// over the top groups -- searched again, or replayed from the cache.
    ///
    /// # Errors
    /// [`Error::IllegalState`] for a search by blocks, or for replaying a
    /// cache without scores into a second pass that reads them (Java's
    /// `NullPointerException`); Java's argument errors; what the search
    /// reports.
    pub fn search<'a, S>(
        &self,
        searcher: &IndexSearcher<'_, 'a>,
        query: &BooleanQuery,
        group_offset: usize,
        group_limit: usize,
    ) -> Result<GroupingSearchResult<S::Value>>
    where
        S: GroupSelector<'a>,
        F: Fn() -> S,
    {
        let Grouper::Selector(factory) = &self.grouper else {
            return Err(Error::IllegalState(
                "Either groupField, groupFunction or groupEndDocs must be set.".into(),
            ));
        };
        let top_n = group_offset.saturating_add(group_limit);
        let mut round = FirstRound {
            first: FirstPassGroupingCollector::new(
                factory(),
                self.group_sort.clone(),
                top_n,
                self.ignore_docs_without_group_field,
            )?,
            all: self.all_groups.then(|| AllGroupsCollector::new(factory())),
            heads: self
                .all_group_heads
                .then(|| AllGroupHeadsCollector::new(factory(), self.sort_within_group.clone())),
        };
        let cache = match (self.max_cache_ram_mb, self.max_docs_to_cache) {
            (Some(mb), _) => Some(CachingCollector::with_ram(
                NoOpCollector,
                self.cache_scores,
                mb,
            )),
            (None, Some(n)) => Some(CachingCollector::with_max_docs(
                NoOpCollector,
                self.cache_scores,
                n,
            )),
            (None, None) => None,
        };
        let cache = match cache {
            Some(mut cache) => {
                let mut driver = PerSegment::new(searcher.segments(), &mut round);
                searcher.search_collector(
                    query,
                    &mut Tee(&mut cache, &mut driver, self.cache_scores),
                )?;
                driver.close()?;
                Some(cache)
            }
            None => {
                search_segments(searcher, query, &mut round)?;
                None
            }
        };
        let max_doc = searcher
            .segments()
            .iter()
            .map(|s| s.max_doc.or(s.reader.map(|r| r.max_doc)).unwrap_or(0))
            .fold(0i32, i32::saturating_add);
        let max_doc = usize::try_from(max_doc).unwrap_or(0);
        let matching_groups = round
            .all
            .as_ref()
            .map_or_else(Vec::new, |a| a.groups().to_vec());
        let matching_group_heads = match &round.heads {
            Some(h) => h.retrieve_group_heads_bits(max_doc),
            None => FixedBitSet::new(max_doc),
        };
        let Some(top_search_groups) = round.first.top_groups(group_offset) else {
            return Ok(GroupingSearchResult {
                top_groups: TopGroups {
                    total_hit_count: 0,
                    total_grouped_hit_count: 0,
                    total_group_count: None,
                    groups: Vec::new(),
                    group_sort: Vec::new(),
                    within_group_sort: Vec::new(),
                    max_score: f32::NAN,
                },
                matching_groups,
                matching_group_heads,
            });
        };
        let top_n_inside_group = self.group_docs_offset.saturating_add(self.group_docs_limit);
        let mut second = TopGroupsCollector::new(
            factory(),
            top_search_groups,
            self.group_sort.clone(),
            self.sort_within_group.clone(),
            top_n_inside_group,
            self.include_max_score,
        )?;
        match cache {
            Some(cache) if cache.is_cached() => {
                if !self.cache_scores {
                    // Java replays without a scorer: a second pass that reads
                    // scores dereferences none (`NullPointerException`), and
                    // the selector is never handed one.
                    if second.score_mode() == ScoreMode::Complete {
                        return Err(Error::IllegalState(
                            "the first pass was cached without scores, and the second pass \
                             reads them"
                                .into(),
                        ));
                    }
                    second.set_has_scorer(false);
                }
                let mut driver = PerSegment::new(searcher.segments(), &mut second);
                cache.replay(&mut driver)?;
                driver.close()?;
            }
            _ => search_segments(searcher, query, &mut second)?,
        }
        let mut top_groups = second.top_groups(self.group_docs_offset);
        if self.all_groups {
            top_groups.total_group_count =
                Some(i32::try_from(matching_groups.len()).unwrap_or(i32::MAX));
        }
        Ok(GroupingSearchResult {
            top_groups,
            matching_groups,
            matching_group_heads,
        })
    }
}

/// `MultiCollector.wrap(firstPassCollector, allGroupsCollector,
/// allGroupHeadsCollector)`: each, in that order, for every segment and
/// document.
struct FirstRound<'a, S: GroupSelector<'a>> {
    first: FirstPassGroupingCollector<'a, S>,
    all: Option<AllGroupsCollector<'a, S>>,
    heads: Option<AllGroupHeadsCollector<'a, S>>,
}

impl<'a, S: GroupSelector<'a>> SegmentCollector<'a> for FirstRound<'a, S> {
    fn score_mode(&self) -> ScoreMode {
        let needs = self.first.score_mode() == ScoreMode::Complete
            || self
                .all
                .as_ref()
                .is_some_and(|c| c.score_mode() == ScoreMode::Complete)
            || self
                .heads
                .as_ref()
                .is_some_and(|c| c.score_mode() == ScoreMode::Complete);
        if needs {
            ScoreMode::Complete
        } else {
            ScoreMode::CompleteNoScores
        }
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.first.set_next_reader(ord, leaf)?;
        // Java's three collectors share one selector, which the first pass
        // hands the scorer; these have their own, handed it here.
        if let Some(c) = self.all.as_mut() {
            c.set_next_reader(ord, leaf)?;
            c.selector_mut().set_scorer()?;
        }
        if let Some(c) = self.heads.as_mut() {
            c.set_next_reader(ord, leaf)?;
            c.selector_mut().set_scorer()?;
        }
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.first.collect(doc, score)?;
        if let Some(c) = self.all.as_mut() {
            c.collect(doc, score)?;
        }
        if let Some(c) = self.heads.as_mut() {
            c.collect(doc, score)?;
        }
        Ok(())
    }
}

/// Two collectors fed the same documents: the cache and the first round
/// (`CachingCollector.create(firstRound, ...)`), asking for scores when the
/// cache keeps them (`ScoreCachingCollector.scoreMode()` is `COMPLETE`) or
/// the first round reads them.
struct Tee<'x, A, B>(&'x mut A, &'x mut B, bool);

impl<A: ScoringCollector, B: ScoringCollector> ScoringCollector for Tee<'_, A, B> {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.0.collect(doc_id, score);
        self.1.collect(doc_id, score);
    }

    fn score_mode(&self) -> ScoreMode {
        if self.2 || self.1.score_mode() == ScoreMode::Complete {
            ScoreMode::Complete
        } else {
            ScoreMode::CompleteNoScores
        }
    }
}
