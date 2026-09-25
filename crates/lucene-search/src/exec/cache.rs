//! `LRUQueryCache` for the scorer tree: a clause built without scores (a
//! `FILTER` or `MUST_NOT` clause, or any clause of a count) that keeps being
//! used is computed once per segment into a doc-id set and iterated from
//! there afterwards.
//!
//! What Lucene does, and this follows: `IndexSearcher.createWeight` wraps
//! every weight created for `COMPLETE_NO_SCORES` in a `CachingWrapperWeight`;
//! `UsageTrackingQueryCachingPolicy` decides from how often a query was seen
//! (costly queries twice, compound ones four times, others five; term,
//! match-all, match-none and empty ones never); a segment under 10,000
//! documents is never cached; the cached set is the segment core's matches,
//! before deletions, so it survives a refresh that only changes live docs; a
//! dense set is a bitset and a sparse one a sorted doc list
//! (`LRUQueryCache.cacheImpl`'s 1% rule).
//!
//! Deviations: the cache belongs to the segment (it lives in
//! [`crate::directory_reader::SegmentReader`] and goes away with it, where
//! Lucene keys one node-wide cache on the core's cache helper), so its
//! bounds are per segment ([`MAX_ENTRIES`], [`MAX_BYTES`]) rather than
//! node-wide; the usage history is per segment too, which counts the same
//! uses since a query visits each segment once; `skipCacheFactor` (don't
//! cache a clause costing ten times its conjunction's lead) and the
//! "at least half the average leaf" size rule are not ported, so a cache
//! entry can be built where Lucene would have run the clause uncached.
//! None of these changes a hit or a score.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use lucene_util::fixed_bit_set::FixedBitSet;

use super::{exact_next, BoxScorer, Scorer, NO_MORE_DOCS};
use crate::query::Clause;
use crate::query_cache::{CachingCost, QueryCachingPolicy, UsageTrackingPolicy};
use crate::Result;

/// Cached queries per segment (Lucene's node-wide default is 1,000).
pub(crate) const MAX_ENTRIES: usize = 64;
/// Cached bytes per segment (Lucene's node-wide default is 32 MB).
pub(crate) const MAX_BYTES: usize = 16 << 20;
/// `LRUQueryCache`'s `MinSegmentSizePredicate` floor.
const MIN_SEGMENT_SIZE: i32 = 10_000;

/// A segment's matches for one query, before deletions.
pub(crate) enum CachedSet {
    Bits { bits: FixedBitSet, cardinality: i64 },
    Docs(Vec<i32>),
}

impl CachedSet {
    fn ram_bytes(&self) -> usize {
        match self {
            CachedSet::Bits { bits, .. } => bits.words().len() * 8,
            CachedSet::Docs(docs) => docs.len() * 4,
        }
    }
}

struct Entry {
    set: Arc<CachedSet>,
    last_used: u64,
}

#[derive(Default)]
struct Inner {
    policy: UsageTrackingPolicy,
    entries: HashMap<String, Entry>,
    clock: u64,
    bytes: usize,
}

/// One segment's query cache. Shared by every search of the segment, hence
/// the lock; held only to look up or insert, never while a set is built.
#[derive(Default)]
pub struct SegmentQueryCache {
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for SegmentQueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentQueryCache").finish_non_exhaustive()
    }
}

/// What the policy is asked about: the clause's identity and shape.
struct PolicyKey<'k> {
    key: &'k str,
    costly: bool,
    composite: bool,
    never: bool,
}

impl Hash for PolicyKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl CachingCost for PolicyKey<'_> {
    fn never_cache(&self) -> bool {
        self.never
    }
    fn is_costly(&self) -> bool {
        self.costly
    }
    fn is_composite(&self) -> bool {
        self.composite
    }
}

/// `UsageTrackingQueryCachingPolicy.shouldNeverCache`, `isCostly` and its
/// compound-query rule, for the clause kinds the tree runs.
fn shape(clause: &Clause) -> (bool, bool, bool) {
    // (costly, composite, never)
    match clause {
        Clause::Term(_) | Clause::MatchAllDocs(_) | Clause::MatchNoDocs(_) => (false, false, true),
        Clause::Boolean(b) => {
            let empty = b.must.is_empty()
                && b.filter.is_empty()
                && b.should.is_empty()
                && b.must_not.is_empty();
            (false, true, empty)
        }
        Clause::DisjunctionMax(d) => (false, true, d.disjuncts.is_empty()),
        Clause::Wildcard(_)
        | Clause::Prefix(_)
        | Clause::Regexp(_)
        | Clause::Fuzzy(_)
        | Clause::TermInSet(_)
        | Clause::PointsRange(_) => (true, false, false),
        _ => (false, false, false),
    }
}

impl SegmentQueryCache {
    /// The cached scorer for `clause` on a segment of `max_doc` documents:
    /// from the cache, or built by `uncached` and stored when the policy
    /// says so. `None` means run `uncached`'s scorer as is.
    pub(crate) fn scorer<'a>(
        &self,
        clause: &Clause,
        max_doc: i32,
        uncached: impl FnOnce() -> Result<Option<BoxScorer<'a>>>,
    ) -> Result<Option<CacheResult<'a>>> {
        let (costly, composite, never) = shape(clause);
        if never || max_doc < MIN_SEGMENT_SIZE {
            return Ok(None);
        }
        let key = format!("{clause:?}");
        let policy_key = PolicyKey {
            key: &key,
            costly,
            composite,
            never,
        };
        let cache = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.clock += 1;
            let now = inner.clock;
            inner.policy.on_use(&policy_key);
            if let Some(entry) = inner.entries.get_mut(&key) {
                entry.last_used = now;
                return Ok(Some(CacheResult::Hit(Box::new(CachedScorer::new(
                    Arc::clone(&entry.set),
                )))));
            }
            inner.policy.should_cache(&policy_key)
        };
        if !cache {
            return Ok(None);
        }
        // Built without the lock: another search may build the same set at
        // the same time, and the second insert simply replaces the first.
        let Some(mut s) = uncached()? else {
            return Ok(Some(CacheResult::Empty));
        };
        let set = Arc::new(collect(&mut *s, max_doc)?);
        self.insert(key, Arc::clone(&set));
        Ok(Some(CacheResult::Hit(Box::new(CachedScorer::new(set)))))
    }

    fn insert(&self, key: String, set: Arc<CachedSet>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.clock += 1;
        let now = inner.clock;
        inner.bytes += set.ram_bytes();
        if let Some(old) = inner.entries.insert(
            key,
            Entry {
                set,
                last_used: now,
            },
        ) {
            inner.bytes -= old.set.ram_bytes();
        }
        // `LRUQueryCache.evictIfNecessary`: least recently used first.
        while inner.entries.len() > MAX_ENTRIES || inner.bytes > MAX_BYTES {
            let Some(oldest) = inner
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(e) = inner.entries.remove(&oldest) {
                inner.bytes -= e.set.ram_bytes();
            }
        }
    }

    /// Cached queries and their bytes, for tests.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        (inner.entries.len(), inner.bytes)
    }
}

pub(crate) enum CacheResult<'a> {
    Hit(BoxScorer<'a>),
    /// The clause matches nothing in this segment.
    Empty,
}

/// `LRUQueryCache.cacheImpl`: every match of `s` (exact, before deletions),
/// as a bitset when at least 1% of the segment matches, a doc list otherwise.
fn collect(s: &mut dyn Scorer, max_doc: i32) -> Result<CachedSet> {
    let len = usize::try_from(max_doc).unwrap_or(0);
    let mut bits = FixedBitSet::new(len);
    let mut cardinality = 0i64;
    let mut doc = exact_next(s)?;
    while doc != NO_MORE_DOCS {
        if let Ok(i) = usize::try_from(doc) {
            if i < len {
                // FBS: `i < len`, the set's length, checked just above.
                bits.set(i);
                cardinality += 1;
            }
        }
        doc = exact_next(s)?;
    }
    if cardinality * 100 >= i64::from(max_doc) {
        return Ok(CachedSet::Bits { bits, cardinality });
    }
    let mut docs = Vec::new();
    let mut at = bits.next_set_bit(0);
    while let Some(i) = at {
        docs.push(i as i32);
        at = bits.next_set_bit(i + 1);
    }
    Ok(CachedSet::Docs(docs))
}

/// A cached set as a scorer: matches only, every score 0.
pub(crate) struct CachedScorer {
    set: Arc<CachedSet>,
    doc: i32,
    /// `Docs`: the index of `doc`.
    at: usize,
}

impl CachedScorer {
    pub(crate) fn new(set: Arc<CachedSet>) -> Self {
        Self {
            set,
            doc: -1,
            at: 0,
        }
    }
}

impl Scorer for CachedScorer {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc.saturating_add(1);
        self.advance(target)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = match &*self.set {
            CachedSet::Bits { bits, .. } => usize::try_from(target)
                .ok()
                .and_then(|t| bits.next_set_bit(t))
                .map_or(NO_MORE_DOCS, |d| d as i32),
            CachedSet::Docs(docs) => {
                // Galloping from the current index: callers advance forward.
                let rest = &docs[self.at..];
                let skip = rest.partition_point(|&d| d < target);
                self.at += skip;
                docs.get(self.at).copied().unwrap_or(NO_MORE_DOCS)
            }
        };
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        match &*self.set {
            CachedSet::Bits { cardinality, .. } => *cardinality,
            CachedSet::Docs(docs) => docs.len() as i64,
        }
    }

    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }

    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }

    fn contains(&self, doc: i32) -> Option<bool> {
        match &*self.set {
            CachedSet::Bits { bits, .. } => {
                Some(usize::try_from(doc).is_ok_and(|d| d < bits.len() && bits.get(d)))
            }
            CachedSet::Docs(_) => None,
        }
    }

    /// `BitSetIterator.docIDRunEnd`: the run of set bits from here.
    fn doc_id_run_end(&self) -> i32 {
        match &*self.set {
            CachedSet::Bits { bits, .. } => {
                let from = usize::try_from(self.doc).unwrap_or(0);
                let end = lucene_util::fixed_bit_set::next_clear_bit_in_words(bits.words(), from);
                end.min(bits.len()) as i32
            }
            CachedSet::Docs(_) => self.doc.saturating_add(1),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::exec::leaf::DocList;
    use crate::query::{
        BooleanQuery, DisjunctionMaxQuery, MatchAllDocsQuery, PhraseQuery, PrefixQuery, TermQuery,
    };

    fn list(docs: Vec<i32>) -> Result<Option<BoxScorer<'static>>> {
        Ok(Some(Box::new(DocList::new(docs, Vec::new()))))
    }

    fn hit(r: Option<CacheResult<'static>>) -> BoxScorer<'static> {
        match r {
            Some(CacheResult::Hit(s)) => s,
            _ => panic!("expected a cached scorer"),
        }
    }

    #[test]
    fn the_policy_shapes_follow_usage_tracking() {
        let term = Clause::Term(TermQuery::new("f", "t"));
        assert_eq!(shape(&term), (false, false, true));
        assert_eq!(
            shape(&Clause::MatchAllDocs(MatchAllDocsQuery::new(3))),
            (false, false, true)
        );
        assert_eq!(
            shape(&Clause::Boolean(Box::new(BooleanQuery::new()))),
            (false, true, true)
        );
        let mut b = BooleanQuery::new();
        b.must.push(term.clone());
        assert_eq!(shape(&Clause::Boolean(Box::new(b))), (false, true, false));
        let empty = DisjunctionMaxQuery::new(Vec::<Clause>::new(), 0.0);
        assert_eq!(
            shape(&Clause::DisjunctionMax(Box::new(empty))),
            (false, true, true)
        );
        assert_eq!(
            shape(&Clause::Prefix(PrefixQuery::new("f", "p"))),
            (true, false, false)
        );
        assert_eq!(
            shape(&Clause::Phrase(PhraseQuery::new("f", ["a", "b"]))),
            (false, false, false)
        );
    }

    #[test]
    fn a_clause_is_cached_after_the_policy_threshold_and_served_from_then_on() {
        let cache = SegmentQueryCache::default();
        assert!(format!("{cache:?}").contains("SegmentQueryCache"));
        let phrase = Clause::Phrase(PhraseQuery::new("f", ["a", "b"]));
        // Never for a small segment, nor for a term.
        assert!(cache
            .scorer(&phrase, 9_999, || list(vec![1]))
            .unwrap()
            .is_none());
        let term = Clause::Term(TermQuery::new("f", "t"));
        assert!(cache
            .scorer(&term, 20_000, || list(vec![1]))
            .unwrap()
            .is_none());
        // A phrase is an ordinary query: cached on its fifth use.
        for _ in 0..4 {
            let r = cache
                .scorer(&phrase, 20_000, || panic!("not cached yet"))
                .unwrap();
            assert!(r.is_none());
        }
        let mut s = hit(cache
            .scorer(&phrase, 20_000, || list(vec![3, 9, 17]))
            .unwrap());
        assert_eq!(cache.stats().0, 1);
        let mut again = hit(cache
            .scorer(&phrase, 20_000, || panic!("served from the cache"))
            .unwrap());
        // Sparse: a doc list.
        for scorer in [&mut s, &mut again] {
            assert_eq!(scorer.cost(), 3);
            assert_eq!(scorer.contains(3), None, "a doc list has no random access");
            assert_eq!(scorer.next_doc().unwrap(), 3);
            assert_eq!(scorer.doc_id_run_end(), 4);
            assert_eq!(scorer.advance(10).unwrap(), 17);
            assert_eq!(scorer.score().unwrap(), 0.0);
            assert_eq!(scorer.max_score(NO_MORE_DOCS).unwrap(), 0.0);
            assert_eq!(scorer.next_doc().unwrap(), NO_MORE_DOCS);
        }
    }

    #[test]
    fn dense_sets_are_bitsets_with_runs_and_empty_ones_match_nothing() {
        let cache = SegmentQueryCache::default();
        let prefix = Clause::Prefix(PrefixQuery::new("f", "p"));
        // Costly: cached on the second use. 300 of 20,000 is over 1%.
        assert!(cache
            .scorer(&prefix, 20_000, || panic!("first use"))
            .unwrap()
            .is_none());
        let dense: Vec<i32> = (100..400).collect();
        let mut s = hit(cache.scorer(&prefix, 20_000, || list(dense)).unwrap());
        assert_eq!(s.cost(), 300);
        assert_eq!(s.next_doc().unwrap(), 100);
        assert_eq!(s.doc_id_run_end(), 400);
        assert_eq!(s.advance(399).unwrap(), 399);
        assert_eq!(s.advance(400).unwrap(), NO_MORE_DOCS);
        let other = Clause::Prefix(PrefixQuery::new("f", "q"));
        assert!(cache
            .scorer(&other, 20_000, || panic!("first use"))
            .unwrap()
            .is_none());
        let r = cache.scorer(&other, 20_000, || Ok(None)).unwrap();
        assert!(matches!(r, Some(CacheResult::Empty)));
    }

    /// Two searches building the same set at once: the second insert
    /// replaces the first, and the byte count stays that of one entry.
    #[test]
    fn a_racing_second_insert_replaces_the_first() {
        let cache = SegmentQueryCache::default();
        let set = || Arc::new(CachedSet::Docs(vec![1, 2, 3]));
        cache.insert("q".to_string(), set());
        cache.insert("q".to_string(), set());
        assert_eq!(cache.stats(), (1, 12));
    }

    #[test]
    fn the_least_recently_used_entries_are_evicted_past_the_bounds() {
        let cache = SegmentQueryCache::default();
        for i in 0..(MAX_ENTRIES + 6) {
            let q = Clause::Prefix(PrefixQuery::new("f", format!("p{i}")));
            assert!(cache
                .scorer(&q, 20_000, || panic!("first use"))
                .unwrap()
                .is_none());
            hit(cache.scorer(&q, 20_000, || list(vec![1, 2])).unwrap());
        }
        let (entries, bytes) = cache.stats();
        assert_eq!(entries, MAX_ENTRIES);
        assert_eq!(bytes, MAX_ENTRIES * 8);
        // The first ones went; the last is still served.
        let last = Clause::Prefix(PrefixQuery::new("f", format!("p{}", MAX_ENTRIES + 5)));
        hit(cache.scorer(&last, 20_000, || panic!("cached")).unwrap());
        let first = Clause::Prefix(PrefixQuery::new("f", "p0"));
        let r = cache.scorer(&first, 20_000, || list(vec![5])).unwrap();
        assert!(
            matches!(r, Some(CacheResult::Hit(_))),
            "rebuilt after eviction"
        );
        // A set over the byte bound evicts everything older, then itself fits
        // alone only if under the bound: 16 MB of bits is 134M documents.
        let huge = Clause::Prefix(PrefixQuery::new("f", "huge"));
        let max_doc = i32::try_from(MAX_BYTES * 8 + 64).unwrap();
        assert!(cache
            .scorer(&huge, max_doc, || panic!("first use"))
            .unwrap()
            .is_none());
        hit(cache
            .scorer(&huge, max_doc, || list((0..max_doc / 50).collect()))
            .unwrap());
        assert!(cache.stats().1 <= MAX_BYTES);
    }
}
