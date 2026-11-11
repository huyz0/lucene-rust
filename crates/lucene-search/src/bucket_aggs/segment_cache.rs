//! A per-segment cache of what an aggregation pass collected from one
//! segment, as Quickwit's partial request cache keeps a split's partial
//! aggregation: a refresh adds segments and leaves the others as they were,
//! so a dashboard's next request reads the unchanged segments' parts back
//! and collects only the new segments. OpenSearch's shard request cache
//! keys a whole shard's result by its reader and starts over on every
//! refresh.
//!
//! **What a part is.** A segment's states as [`super::aggregate_tree`]
//! leaves them after the segment's last document, before `finish`: each
//! keyed bucket's key and count, in the order the segment created them,
//! each fixed bucket's count, each bucket's metric state, and the same for
//! the sub-aggregations under each bucket. A keyword `terms`' keys are kept
//! as the segment's own ordinals, which a reader with other segments maps
//! to its global ones. Adding a part to a slice's states makes the buckets
//! a sequential collection of the segment would have made, in the order it
//! would have made them, so `finish` lists and truncates exactly as it does
//! without the cache.
//!
//! **What a part is keyed by.** The segment's identity
//! ([`SegmentIdentity`]: its id, name, size, each field's doc-values
//! generation but the soft-deletes field's, and its live documents), the
//! tree, and the query with each points range narrowed to the segment's own
//! values. A `range` from
//! `now-24h` moves with every request, but on a segment whose values all
//! fall inside it the narrowed range is the segment's own bounds, so the
//! key stays the same until the segment changes. A deletion changes the
//! live documents, a doc-values update a generation, and a merge the
//! segment: each makes a new key, and the old entry ages out.
//!
//! **Blocks.** A merged segment spans much of the index's time, so a
//! moving range's end usually falls inside one and its key would change
//! with every request. A segment that a range cuts, bigger than a block
//! (2^16 documents), is cached in blocks of consecutive documents nested
//! four to one, each keyed by its own live documents and the query narrowed
//! to its own values of each cut field (read once from the field's points):
//! a block the ranges cover or miss is one entry, a block they cut is its
//! four children, down to the smallest. Log documents arrive in time order,
//! so a block's values span a short time: a range moving through a segment
//! keeps the keys of the big blocks it covers and collects at most the
//! smallest blocks its ends fall in. The blocks' parts make the buckets the
//! segment would have made, if not in the same order, and `finish` orders
//! buckets by their counts and keys alone.
//!
//! **What is cached.** `terms` (keyword, without `missing` or
//! `include`/`exclude`), numeric `terms`, `histogram`, `date_histogram`,
//! `range`, `filter`/`filters`, and the metrics under them, over queries of
//! terms, ranges, prefixes, wildcards, phrases, `exists` and booleans of
//! them, without scores. A metric summing doubles is not: a sum's rounding
//! depends on the order of its values, which the cache changes. A sum of
//! whole numbers is exact while it stays below 2^53; each part keeps its
//! values' bounds, and a part that could round when added makes the pass
//! run again without the cache. Anything else runs without it.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use lucene_util::fixed_bit_set::FixedBitSet;

use super::{keyed_ord, slot, AggNode, Globals, State};
use crate::aggs::{self, MetricState, ValueKind};
use crate::directory_reader::SegmentReader;
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause};
use crate::Result;

/// Exact sums of whole numbers stay below this.
const EXACT: f64 = 9_007_199_254_740_992.0;

/// A segment that a range cuts is cached in blocks of at least this many
/// documents ([`AggSegmentCache::with_block_docs`]).
const BLOCK_DOCS: i32 = 1 << 16;

/// Each block of a cut segment is four blocks of the next size down.
const FANOUT: i32 = 4;

/// At most this many segments' trees' last live documents are kept.
const SEEN_KEPT: usize = 1 << 16;

/// At most this many segments' block bounds are kept.
const BOUNDS_KEPT: usize = 4096;

/// A part bigger than this share of the budget is not kept: one segment's
/// part would push out every other.
const LARGEST_SHARE: u64 = 8;

/// A cached part could not be added in exactly as collecting its segment
/// would have added its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inexact;

/// A segment's identity for the cache: what its documents and their values
/// are, as far as an aggregation reads them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentIdentity {
    /// The segment with its live documents.
    whole: u128,
    /// The segment's documents and values, whichever are deleted: a block's
    /// key adds the block's own live documents.
    core: u128,
}

impl SegmentIdentity {
    /// `reader`'s segment with the live documents `live` (`None`: none
    /// deleted), which replace the segment's own as a search reads them.
    pub fn of(reader: &SegmentReader, live: Option<&FixedBitSet>) -> Self {
        let mut h = Hash128::new();
        reader.segment_id().hash(&mut h);
        reader.segment_name.hash(&mut h);
        reader.max_doc.hash(&mut h);
        // Each field's doc-values generation, a doc-values update's mark,
        // but the soft-deletes field's: a soft delete or an update shows in
        // the live documents, and leaves the other fields' values as they
        // were.
        for f in reader
            .field_infos()
            .fields
            .iter()
            .filter(|f| !f.soft_deletes_field)
        {
            (&f.name, f.number, f.doc_values_gen).hash(&mut h);
        }
        let core = h.finish128();
        hash_live(live, 0, reader.max_doc, &mut h);
        SegmentIdentity {
            whole: h.finish128(),
            core,
        }
    }
}

/// The live documents of `lo..hi` (a whole number of words from `lo`).
fn hash_live(live: Option<&FixedBitSet>, lo: i32, hi: i32, h: &mut Hash128) {
    match live {
        None => 0u8.hash(h),
        Some(l) => {
            1u8.hash(h);
            l.len().hash(h);
            let words = l.words();
            let from = usize::try_from(lo / 64).unwrap_or(0).min(words.len());
            let to = usize::try_from((hi + 63) / 64)
                .unwrap_or(0)
                .clamp(from, words.len());
            words[from..to].hash(h);
        }
    }
}

/// Two SipHashes keyed apart: a 128-bit key, so two different requests
/// never share an entry in practice.
struct Hash128(
    std::collections::hash_map::DefaultHasher,
    std::collections::hash_map::DefaultHasher,
);

impl Hash128 {
    fn new() -> Self {
        let mut b = std::collections::hash_map::DefaultHasher::new();
        b.write_u8(0x5a);
        Hash128(std::collections::hash_map::DefaultHasher::new(), b)
    }

    fn finish128(&self) -> u128 {
        (u128::from(self.0.finish()) << 64) | u128::from(self.1.finish())
    }
}

impl Hasher for Hash128 {
    fn finish(&self) -> u64 {
        self.0.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
        self.1.write(bytes);
    }
}

/// What the cache has done: lookups answered and not, parts dropped for
/// room, and what it holds now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: u64,
    pub bytes: u64,
    pub budget: u64,
}

/// The parts, least recently used dropped first once their bytes pass the
/// budget. A budget of 0 turns the cache off.
pub struct AggSegmentCache {
    budget: AtomicU64,
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    block_docs: i32,
}

/// Each block's smallest and largest value of a field.
type BlockBounds = Vec<(i64, i64)>;

#[derive(Default)]
struct Inner {
    map: HashMap<u128, Slot>,
    /// Each cut segment's blocks' smallest and largest value of a points
    /// field, by the segment's core identity and the field.
    bounds: HashMap<(u128, String), Arc<BlockBounds>>,
    /// The live documents each segment had the last time each tree read
    /// it, by the segment's core identity and the tree.
    seen: HashMap<(u128, u128), u128>,
    bytes: u64,
    tick: u64,
}

struct Slot {
    part: Arc<SegmentPart>,
    bytes: u64,
    used: u64,
}

impl AggSegmentCache {
    pub fn new(budget: u64) -> Self {
        Self::with_block_docs(budget, BLOCK_DOCS)
    }

    /// A cache keeping a segment that a range cuts in blocks of `docs`
    /// documents (rounded up to whole words of live documents), so that a
    /// range moving through a segment collects only the blocks at its ends.
    pub fn with_block_docs(budget: u64, docs: i32) -> Self {
        AggSegmentCache {
            budget: AtomicU64::new(budget),
            inner: Mutex::new(Inner::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            block_docs: (docs.clamp(1, i32::MAX - 63) + 63) & !63,
        }
    }

    /// The process's cache, off until [`Self::set_budget`] gives it room.
    pub fn global() -> &'static AggSegmentCache {
        static CACHE: OnceLock<AggSegmentCache> = OnceLock::new();
        CACHE.get_or_init(|| AggSegmentCache::new(0))
    }

    /// The bytes the parts may take; parts past a smaller budget are dropped
    /// now.
    pub fn set_budget(&self, bytes: u64) {
        self.budget.store(bytes, Ordering::Relaxed);
        let mut inner = self.lock();
        self.evict(&mut inner, bytes);
    }

    pub fn budget(&self) -> u64 {
        self.budget.load(Ordering::Relaxed)
    }

    /// Drops every part.
    pub fn clear(&self) {
        let mut inner = self.lock();
        inner.map.clear();
        inner.bounds.clear();
        inner.seen.clear();
        inner.bytes = 0;
    }

    /// Whether the segment `id`'s live documents are those `tree` last read
    /// it with, or it never read it: a segment whose deletions change
    /// between a tree's requests (updates spread over the index) would miss
    /// on every request and pay for parts never read back, so it is
    /// collected without the cache until two requests find it unchanged.
    fn settled(&self, id: SegmentIdentity, tree: u128) -> bool {
        let mut inner = self.lock();
        if inner.seen.len() >= SEEN_KEPT {
            inner.seen.clear();
        }
        match inner.seen.insert((id.core, tree), id.whole) {
            Some(last) => last == id.whole,
            None => true,
        }
    }

    /// `field`'s bounds in each block of the segment `core`, read from its
    /// points the first time.
    fn block_bounds(
        &self,
        core: u128,
        field: &str,
        points: &crate::points_query::PointsInput<'_>,
        max_doc: i32,
    ) -> Option<Arc<BlockBounds>> {
        let key = (core, field.to_string());
        if let Some(b) = self.lock().bounds.get(&key) {
            return Some(Arc::clone(b));
        }
        let b = Arc::new(read_block_bounds(points, field, self.block_docs, max_doc)?);
        let mut inner = self.lock();
        if inner.bounds.len() >= BOUNDS_KEPT {
            inner.bounds.clear();
        }
        inner.bounds.insert(key, Arc::clone(&b));
        Some(b)
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.lock();
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            entries: inner.map.len() as u64,
            bytes: inner.bytes,
            budget: self.budget(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn get(&self, key: u128) -> Option<Arc<SegmentPart>> {
        let mut inner = self.lock();
        inner.tick += 1;
        let tick = inner.tick;
        let found = inner.map.get_mut(&key).map(|s| {
            s.used = tick;
            Arc::clone(&s.part)
        });
        drop(inner);
        let counter = if found.is_some() {
            &self.hits
        } else {
            &self.misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
        found
    }

    fn put(&self, key: u128, part: SegmentPart) -> Arc<SegmentPart> {
        let part = Arc::new(part);
        let bytes = part.bytes;
        let budget = self.budget();
        if bytes.saturating_mul(LARGEST_SHARE) > budget {
            return part;
        }
        let mut inner = self.lock();
        inner.tick += 1;
        let used = inner.tick;
        let old = inner.map.insert(
            key,
            Slot {
                part: Arc::clone(&part),
                bytes,
                used,
            },
        );
        inner.bytes = inner
            .bytes
            .saturating_add(bytes)
            .saturating_sub(old.map_or(0, |o| o.bytes));
        if inner.bytes > budget {
            // Down to seven eighths, so that the next few parts do not each
            // pay for a scan.
            self.evict(&mut inner, budget - budget / 8);
        }
        part
    }

    /// Drops the least recently used parts until the rest fit in `room`.
    fn evict(&self, inner: &mut Inner, room: u64) {
        if inner.bytes <= room {
            return;
        }
        let mut order: Vec<(u64, u128)> = inner.map.iter().map(|(k, s)| (s.used, *k)).collect();
        order.sort_unstable();
        for (_, k) in order {
            if inner.bytes <= room {
                break;
            }
            if let Some(s) = inner.map.remove(&k) {
                inner.bytes = inner.bytes.saturating_sub(s.bytes);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// A pass's cache, and each segment's identity in it: `None` for a segment
/// not to cache (a realtime buffer's, which the next view replaces).
pub struct CacheRequest<'a> {
    pub cache: &'a AggSegmentCache,
    pub segments: &'a [Option<SegmentIdentity>],
}

/// The part of a key the request's segments share.
pub(super) struct RequestKey<'a> {
    request: &'a CacheRequest<'a>,
    tree: u128,
}

impl<'a> RequestKey<'a> {
    /// `None` when the cache is off or cannot answer the tree or query.
    pub(super) fn new(
        request: &'a CacheRequest<'a>,
        query: &BooleanQuery,
        nodes: &[AggNode],
    ) -> Option<Self> {
        if request.cache.budget() == 0 || !nodes_cacheable(nodes) || !boolean_cacheable(query) {
            return None;
        }
        let mut h = Hash128::new();
        format!("{nodes:?}").hash(&mut h);
        Some(RequestKey {
            request,
            tree: h.finish128(),
        })
    }

    /// Segment `i`'s entries for `query`, each with the documents it
    /// covers in order, `None` for a segment not to cache. One entry covers
    /// the segment, unless a points range cuts it and it is bigger than a
    /// block: then each block has its own entry, its ranges narrowed to the
    /// block's values, so that a range moving through the segment finds every
    /// block it covers, and a deletion in one block leaves the others.
    pub(super) fn entries(
        &self,
        i: usize,
        query: &BooleanQuery,
        seg: &OpenSegment<'_>,
        max_doc: i32,
    ) -> Option<Vec<(i32, i32, Entry<'a>)>> {
        let id = self.request.segments.get(i).copied().flatten()?;
        let cache = self.request.cache;
        if !cache.settled(id, self.tree) {
            return None;
        }
        let segment_bounds = |field: &str| seg.points.and_then(|p| points_bounds(p, field));
        let mut cut = Vec::new();
        let narrowed = seg
            .points
            .filter(|_| has_range(query))
            .map(|p| narrow(query, p, &segment_bounds, &mut cut));
        let points = match seg.points {
            Some(p) if !cut.is_empty() && max_doc > cache.block_docs => p,
            _ => {
                let mut h = Hash128::new();
                id.whole.hash(&mut h);
                self.tree.hash(&mut h);
                format!("{:?}", narrowed.as_ref().unwrap_or(query)).hash(&mut h);
                let entry = Entry {
                    cache,
                    key: h.finish128(),
                };
                return Some(vec![(0, max_doc, entry)]);
            }
        };
        let mut tables = Vec::with_capacity(cut.len());
        for field in cut {
            let table = cache.block_bounds(id.core, &field, points, max_doc)?;
            tables.push((field, table));
        }
        // Blocks nested four to one from the smallest up to the segment:
        // a block whose ranges cover it or miss it is one entry, a block a
        // range cuts is its four children, down to the smallest blocks.
        let leaf = cache.block_docs;
        let mut size = leaf;
        while size < max_doc {
            size = size.saturating_mul(FANOUT);
        }
        let mut out = Vec::new();
        let mut stack = vec![(0i32, size)];
        while let Some((lo, size)) = stack.pop() {
            let hi = lo.saturating_add(size).min(max_doc);
            let leaves = usize::try_from(lo / leaf).unwrap_or(0)
                ..usize::try_from((hi + leaf - 1) / leaf).unwrap_or(0);
            let bounds = |field: &str| {
                tables
                    .iter()
                    .find(|(f, _)| f == field)
                    .and_then(|(_, t)| {
                        t.get(leaves.clone())?
                            .iter()
                            .copied()
                            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
                    })
                    .or_else(|| segment_bounds(field))
            };
            let mut cut = Vec::new();
            let q = narrow(query, points, &bounds, &mut cut);
            if !cut.is_empty() && size > leaf && size < i32::MAX {
                let child = size / FANOUT;
                for c in (0..FANOUT).rev() {
                    let from = lo.saturating_add(c.saturating_mul(child));
                    if from < hi {
                        stack.push((from, child));
                    }
                }
                continue;
            }
            let mut h = Hash128::new();
            id.core.hash(&mut h);
            (lo, hi).hash(&mut h);
            hash_live(seg.live_docs, lo, hi, &mut h);
            self.tree.hash(&mut h);
            format!("{q:?}").hash(&mut h);
            out.push((
                lo,
                hi,
                Entry {
                    cache,
                    key: h.finish128(),
                },
            ));
        }
        Some(out)
    }
}

/// One segment's slot in the cache.
pub(super) struct Entry<'a> {
    cache: &'a AggSegmentCache,
    key: u128,
}

impl Entry<'_> {
    pub(super) fn get(&self) -> Option<Arc<SegmentPart>> {
        self.cache.get(self.key)
    }

    pub(super) fn put(&self, part: SegmentPart) -> Arc<SegmentPart> {
        self.cache.put(self.key, part)
    }
}

/// Whether every node, and every node under it, can be cached.
fn nodes_cacheable(nodes: &[AggNode]) -> bool {
    nodes.iter().all(|n| {
        let own = match n {
            AggNode::Metric {
                script,
                kind,
                needs,
                ..
            } => {
                script.is_none()
                    && (*kind == ValueKind::Long || collected(*needs) & aggs::NEED_SUM == 0)
            }
            AggNode::Terms {
                missing, accepted, ..
            } => missing.is_none() && accepted.is_none(),
            AggNode::NumericTerms { .. }
            | AggNode::Histogram { .. }
            | AggNode::DateHistogram { .. }
            | AggNode::Range { .. } => true,
            AggNode::Filters { filters, .. } => filters.iter().all(boolean_cacheable),
            _ => false,
        };
        own && nodes_cacheable(n.subs())
    })
}

/// Whether a query's matches on a segment depend on that segment alone,
/// and its debug form spells out everything they depend on.
fn boolean_cacheable(q: &BooleanQuery) -> bool {
    q.must
        .iter()
        .chain(&q.filter)
        .chain(&q.should)
        .chain(&q.must_not)
        .all(clause_cacheable)
}

fn clause_cacheable(c: &Clause) -> bool {
    use crate::extended_query::ExtendedQuery;
    match c {
        Clause::Term(_)
        | Clause::TermInSet(_)
        | Clause::Phrase(_)
        | Clause::PointsRange(_)
        | Clause::MatchAllDocs(_)
        | Clause::MatchNoDocs(_)
        | Clause::Exists(_)
        | Clause::Prefix(_)
        | Clause::Wildcard(_) => true,
        Clause::Boolean(b) => boolean_cacheable(b),
        Clause::ConstantScore(cs) => clause_cacheable(&cs.inner),
        Clause::Boost(b) => clause_cacheable(&b.inner),
        Clause::Extended(e) => matches!(
            **e,
            ExtendedQuery::PointRange(_) | ExtendedQuery::PointInSet(_)
        ),
        _ => false,
    }
}

/// A one-dimensional eight-byte points field's smallest and largest value
/// in the segment, as the sortable longs a points range compares.
fn points_bounds(points: &crate::points_query::PointsInput<'_>, field: &str) -> Option<(i64, i64)> {
    let n = points.field_number(field)?;
    let f = points.reader.field(n)?;
    if f.num_dims != 1 || f.bytes_per_dim != 8 {
        return None;
    }
    Some((
        unpack(f.min_packed_value.get(..8)?)?,
        unpack(f.max_packed_value.get(..8)?)?,
    ))
}

fn unpack(b: &[u8]) -> Option<i64> {
    let b: [u8; 8] = b.try_into().ok()?;
    Some((u64::from_be_bytes(b) ^ 0x8000_0000_0000_0000) as i64)
}

/// Each block's smallest and largest value of `field`, from every point of
/// the segment, deleted or not; a block without a value has an empty range.
fn read_block_bounds(
    points: &crate::points_query::PointsInput<'_>,
    field: &str,
    block_docs: i32,
    max_doc: i32,
) -> Option<BlockBounds> {
    use lucene_codecs::points::Relation;
    struct Visitor {
        block_docs: i32,
        blocks: Vec<(i64, i64)>,
    }
    impl lucene_codecs::points::IntersectVisitor for Visitor {
        fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
            Relation::CellCrossesQuery
        }
        fn visit(&mut self, _doc_id: i32) {}
        fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
            let (Some(v), Ok(b)) = (
                packed_value.get(..8).and_then(unpack),
                usize::try_from(doc_id / self.block_docs),
            ) else {
                return;
            };
            if let Some(slot) = self.blocks.get_mut(b) {
                slot.0 = slot.0.min(v);
                slot.1 = slot.1.max(v);
            }
        }
    }
    points_bounds(points, field)?;
    let n = points.field_number(field)?;
    let count = usize::try_from(max_doc.div_euclid(block_docs) + 1).ok()?;
    let mut v = Visitor {
        block_docs,
        blocks: vec![(i64::MAX, i64::MIN); count],
    };
    points.reader.intersect(n, &mut v).ok()?;
    Some(v.blocks)
}

/// `query` with each points range narrowed to the values `bounds` gives for
/// its field (no value in it: a match-none), so that a range covering the
/// documents keys them the same wherever its ends are. Each field whose
/// range cuts its bounds is pushed onto `cut`.
fn narrow(
    query: &BooleanQuery,
    points: &crate::points_query::PointsInput<'_>,
    bounds: &dyn Fn(&str) -> Option<(i64, i64)>,
    cut: &mut Vec<String>,
) -> BooleanQuery {
    fn clause(
        c: &Clause,
        points: &crate::points_query::PointsInput<'_>,
        bounds: &dyn Fn(&str) -> Option<(i64, i64)>,
        cut: &mut Vec<String>,
    ) -> Clause {
        match c {
            Clause::PointsRange(r) => {
                if points.field_number(&r.field).is_none() {
                    return Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new());
                }
                match bounds(&r.field) {
                    Some((lo, hi)) if r.max < lo || r.min > hi => {
                        Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new())
                    }
                    Some((lo, hi)) => {
                        if (r.min > lo || r.max < hi) && !cut.contains(&r.field) {
                            cut.push(r.field.clone());
                        }
                        let mut r = r.clone();
                        r.min = r.min.max(lo);
                        r.max = r.max.min(hi);
                        Clause::PointsRange(r)
                    }
                    None => c.clone(),
                }
            }
            Clause::Boolean(b) => Clause::Boolean(Box::new(boolean(b, points, bounds, cut))),
            Clause::ConstantScore(cs) => {
                let mut cs = (**cs).clone();
                cs.inner = Box::new(clause(&cs.inner, points, bounds, cut));
                Clause::ConstantScore(Box::new(cs))
            }
            Clause::Boost(b) => {
                let mut b = (**b).clone();
                b.inner = Box::new(clause(&b.inner, points, bounds, cut));
                Clause::Boost(Box::new(b))
            }
            _ => c.clone(),
        }
    }
    fn boolean(
        b: &BooleanQuery,
        points: &crate::points_query::PointsInput<'_>,
        bounds: &dyn Fn(&str) -> Option<(i64, i64)>,
        cut: &mut Vec<String>,
    ) -> BooleanQuery {
        let mut out = b.clone();
        for list in [
            &mut out.must,
            &mut out.filter,
            &mut out.should,
            &mut out.must_not,
        ] {
            for c in list.iter_mut() {
                *c = clause(c, points, bounds, cut);
            }
        }
        out
    }
    boolean(query, points, bounds, cut)
}

fn has_range(q: &BooleanQuery) -> bool {
    fn clause(c: &Clause) -> bool {
        match c {
            Clause::PointsRange(_) => true,
            Clause::Boolean(b) => has_range(b),
            Clause::ConstantScore(cs) => clause(&cs.inner),
            Clause::Boost(b) => clause(&b.inner),
            _ => false,
        }
    }
    q.must
        .iter()
        .chain(&q.filter)
        .chain(&q.should)
        .chain(&q.must_not)
        .any(clause)
}

/// The tree a cached pass collects with: a whole-number sum also keeps its
/// values' count and bounds, which tell whether its part adds in exactly.
/// The caller reads the parts its nodes asked for, as before.
pub(super) fn collecting(nodes: &[AggNode]) -> Vec<AggNode> {
    nodes
        .iter()
        .map(|n| {
            let mut n = n.clone();
            augment(&mut n);
            n
        })
        .collect()
}

fn augment(n: &mut AggNode) {
    match n {
        AggNode::Metric { needs, .. } => {
            if collected(*needs) & aggs::NEED_SUM != 0 {
                *needs = aggs::NEED_ALL;
            }
        }
        AggNode::Terms { subs, .. }
        | AggNode::NumericTerms { subs, .. }
        | AggNode::Histogram { subs, .. }
        | AggNode::DateHistogram { subs, .. }
        | AggNode::Range { subs, .. }
        | AggNode::Filters { subs, .. } => subs.iter_mut().for_each(augment),
        _ => {}
    }
}

/// Puts back what [`collecting`] added to the metrics the caller did not
/// ask for, so the results are those of the tree as asked.
pub(super) fn restore(asked: &[AggNode], results: &mut [super::AggResult]) {
    use super::AggResult;
    for (n, r) in asked.iter().zip(results.iter_mut()) {
        match (n, r) {
            (AggNode::Metric { needs, .. }, AggResult::Metric(states)) => {
                let kept = collected(*needs);
                if collected(aggs::NEED_ALL) == kept || kept & aggs::NEED_SUM == 0 {
                    continue;
                }
                let fresh = MetricState::default();
                for s in states {
                    if kept & aggs::NEED_MIN_MAX == 0 {
                        s.min = fresh.min;
                        s.max = fresh.max;
                    }
                    if kept & aggs::NEED_MIN == 0 {
                        s.min_of_mins = fresh.min_of_mins;
                    }
                    if kept & aggs::NEED_MAX == 0 {
                        s.max_of_maxes = fresh.max_of_maxes;
                    }
                }
            }
            (
                _,
                AggResult::Terms { subs, .. }
                | AggResult::NumericTerms { subs, .. }
                | AggResult::Histogram { subs, .. }
                | AggResult::DateHistogram { subs, .. }
                | AggResult::Fixed { subs, .. },
            ) => restore(n.subs(), subs),
            _ => {}
        }
    }
}

/// The parts a pass keeps of a metric asked for `needs`
/// ([`MetricState::collect_needs`]'s dispatch, which the windows share):
/// one extreme or the count alone, the count and sum together, or every
/// part.
fn collected(needs: u8) -> u8 {
    match needs {
        aggs::NEED_MIN | aggs::NEED_MAX | aggs::NEED_COUNT => needs,
        n if n & !(aggs::NEED_COUNT | aggs::NEED_SUM) == 0 => aggs::NEED_COUNT | aggs::NEED_SUM,
        _ => aggs::NEED_ALL,
    }
}

/// One segment's collected states ([`super::State`] of the cacheable
/// kinds), keyword keys as the segment's own ordinals.
pub(super) struct SegmentPart {
    parts: Vec<Part>,
    bytes: u64,
}

enum Part {
    /// Per owning bucket.
    Metric(Vec<MetricState>),
    /// Each bucket's `(owning bucket, key)` in creation order, and count.
    Keyed {
        keys: Vec<(u32, i64)>,
        docs: Vec<u64>,
        subs: Vec<Part>,
    },
    /// `owning * width + i`'s count.
    Fixed { docs: Vec<u64>, subs: Vec<Part> },
}

impl Part {
    fn bytes(&self) -> u64 {
        let own = match self {
            Part::Metric(v) => v.len() * std::mem::size_of::<MetricState>(),
            Part::Keyed { keys, docs, .. } => keys.len() * 16 + docs.len() * 8,
            Part::Fixed { docs, .. } => docs.len() * 8,
        };
        let subs: u64 = match self {
            Part::Metric(_) => 0,
            Part::Keyed { subs, .. } | Part::Fixed { subs, .. } => {
                subs.iter().map(Part::bytes).sum()
            }
        };
        own as u64 + subs + 48
    }
}

impl SegmentPart {
    /// Segment `i`'s `states`, just collected.
    pub(super) fn of(
        nodes: &[AggNode],
        states: Vec<State>,
        i: usize,
        globals: &Globals<'_>,
    ) -> Result<Self> {
        let parts = parts_of(nodes, states, i, globals)?;
        let bytes = parts.iter().map(Part::bytes).sum::<u64>() + 64;
        Ok(SegmentPart { parts, bytes })
    }

    /// Adds this part, segment `i`'s, to a slice's `states`; `Ok(false)`
    /// when a sum would not add in exactly (the states are then unusable).
    pub(super) fn add_to(
        &self,
        nodes: &[AggNode],
        states: &mut [State],
        i: usize,
        globals: &Globals<'_>,
    ) -> Result<bool> {
        for ((n, s), p) in nodes.iter().zip(states).zip(&self.parts) {
            if !add(n, s, p, &[0], i, globals)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn mismatch() -> crate::Error {
    crate::Error::TermsAggType("aggregation state does not match its node".to_string())
}

fn parts_of(
    nodes: &[AggNode],
    states: Vec<State>,
    i: usize,
    globals: &Globals<'_>,
) -> Result<Vec<Part>> {
    nodes
        .iter()
        .zip(states)
        .map(|(n, s)| part_of(n, s, i, globals))
        .collect()
}

fn part_of(node: &AggNode, state: State, i: usize, globals: &Globals<'_>) -> Result<Part> {
    match (node, state) {
        (AggNode::Metric { .. }, State::Metric(v)) => Ok(Part::Metric(v)),
        (
            AggNode::Terms { field, .. },
            State::Keyed {
                mut keys,
                docs,
                subs,
                ..
            },
        ) => {
            // Global ordinals back to the segment's: its map is ascending.
            let map = globals.get(field)?.segment_map(i).unwrap_or(&[]);
            for k in &mut keys {
                let ord = map.binary_search(&k.1).map_err(|_| {
                    crate::Error::TermsAggType(format!(
                        "{field}: global ordinal {} is not in segment {i}",
                        k.1
                    ))
                })?;
                k.1 = ord as i64;
            }
            Ok(Part::Keyed {
                keys,
                docs,
                subs: parts_of(node.subs(), subs, i, globals)?,
            })
        }
        (
            AggNode::NumericTerms { .. }
            | AggNode::Histogram { .. }
            | AggNode::DateHistogram { .. },
            State::Keyed {
                keys, docs, subs, ..
            },
        ) => Ok(Part::Keyed {
            keys,
            docs,
            subs: parts_of(node.subs(), subs, i, globals)?,
        }),
        (AggNode::Range { .. } | AggNode::Filters { .. }, State::Fixed { docs, subs, .. }) => {
            Ok(Part::Fixed {
                docs,
                subs: parts_of(node.subs(), subs, i, globals)?,
            })
        }
        _ => Err(mismatch()),
    }
}

/// Adds `part` to `state`, the part's owning bucket `o` being the state's
/// `owners[o]`.
fn add(
    node: &AggNode,
    state: &mut State,
    part: &Part,
    owners: &[u32],
    i: usize,
    globals: &Globals<'_>,
) -> Result<bool> {
    match (node, state, part) {
        (AggNode::Metric { needs, .. }, State::Metric(dst), Part::Metric(src)) => {
            let sums = collected(*needs) & aggs::NEED_SUM != 0;
            for (o, s) in src.iter().enumerate() {
                let Some(&to) = owners.get(o) else {
                    return Err(mismatch());
                };
                if !add_metric(slot(dst, to as usize), s, sums) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (
            _,
            State::Keyed {
                ords,
                keys,
                docs,
                subs,
            },
            Part::Keyed {
                keys: src_keys,
                docs: src_docs,
                subs: src_subs,
            },
        ) => {
            let map = match node {
                AggNode::Terms { field, .. } => {
                    Some(globals.get(field)?.segment_map(i).unwrap_or(&[]))
                }
                _ => None,
            };
            let mut child = Vec::with_capacity(src_keys.len());
            for (b, &(o, k)) in src_keys.iter().enumerate() {
                let Some(&to) = owners.get(o as usize) else {
                    return Err(mismatch());
                };
                let key = match map {
                    Some(m) => *usize::try_from(k)
                        .ok()
                        .and_then(|k| m.get(k))
                        .ok_or_else(mismatch)?,
                    None => k,
                };
                let ord = keyed_ord(ords, keys, to, key);
                if let Some(&n) = src_docs.get(b) {
                    *slot(docs, ord as usize) += n;
                }
                child.push(ord);
            }
            add_subs(node, subs, src_subs, &child, i, globals)
        }
        (
            _,
            State::Fixed {
                width, docs, subs, ..
            },
            Part::Fixed {
                docs: src_docs,
                subs: src_subs,
            },
        ) => {
            let width = *width;
            let mut child = Vec::with_capacity(owners.len() * width);
            for (o, &to) in owners.iter().enumerate() {
                for w in 0..width {
                    let from = o * width + w;
                    let into = to as usize * width + w;
                    if let Some(&n) = src_docs.get(from) {
                        *slot(docs, into) += n;
                    }
                    child.push(into as u32);
                }
            }
            add_subs(node, subs, src_subs, &child, i, globals)
        }
        _ => Err(mismatch()),
    }
}

fn add_subs(
    node: &AggNode,
    states: &mut [State],
    parts: &[Part],
    owners: &[u32],
    i: usize,
    globals: &Globals<'_>,
) -> Result<bool> {
    for ((n, s), p) in node.subs().iter().zip(states).zip(parts) {
        if !add(n, s, p, owners, i, globals)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `dst` as if `src`'s values had been collected into it after its own;
/// false when a sum could round differently.
fn add_metric(dst: &mut MetricState, src: &MetricState, sums: bool) -> bool {
    if sums && src.count > 0 {
        // Every value of the part is a whole number of at most `largest`,
        // so every partial sum a sequential collection would have made
        // stays within `bound` of `dst.sum`: below 2^53 each is exact, and
        // the order of the additions cannot show.
        let largest = src.min.abs().max(src.max.abs());
        let bound = src.count as f64 * largest;
        if !(dst.delta == 0.0
            && src.delta == 0.0
            && bound.is_finite()
            && dst.sum.abs() + bound < EXACT)
        {
            return false;
        }
        dst.sum += src.sum;
    }
    dst.count += src.count;
    dst.min = aggs::java_min(dst.min, src.min);
    dst.max = aggs::java_max(dst.max, src.max);
    dst.min_of_mins = aggs::java_min(dst.min_of_mins, src.min_of_mins);
    dst.max_of_maxes = aggs::java_max(dst.max_of_maxes, src.max_of_maxes);
    true
}
