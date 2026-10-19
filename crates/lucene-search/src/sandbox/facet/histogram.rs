//! `org.apache.lucene.sandbox.facet.plain.histograms`: the number of
//! matching documents per fixed-width bucket of a numeric doc-values field
//! (`HistogramCollectorManager`, `HistogramCollector`,
//! `PointTreeBulkCollector`).
//!
//! Java's collector reads the search's `Weight` to take a shortcut: on a
//! segment the query matches whole, or for a point range on the same field,
//! it counts the field's BKD tree instead of the documents (deletions not
//! applied, as in Java). The Rust collector is handed the query by
//! [`histogram_search`], which stands in for `searcher.search(query,
//! manager)`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use lucene_codecs::doc_values::DocValuesSkipper;
use lucene_codecs::field_infos::DocValuesType;
use lucene_codecs::points::IntersectVisitor;
use lucene_util::numeric_utils::{sortable_bytes_to_int, sortable_bytes_to_long};
use lucene_util::point_values_relation::Relation;

use crate::collector::ScoreMode;
use crate::extended_query::ExtendedQuery;
use crate::grouping::{search_manager, GroupingCollectorManager};
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::SegmentCollector;
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause};
use crate::reader::doc_values::{doc_values_type_name, get_sorted_numeric};
use crate::reader::{IndexReader, LeafReader, PointValues, SortedNumericDocValues};
use crate::terminate::{leaf_count, LeafCount};
use crate::{Error, Result};

use super::FastMap;

/// `HistogramCollectorManager.DEFAULT_MAX_BUCKETS`.
pub const DEFAULT_MAX_BUCKETS: i32 = 1024;

/// `HistogramCollectorManager`: a field's bucket counts, at most
/// `max_buckets` of them.
#[derive(Debug, Clone)]
pub struct HistogramCollectorManager {
    field: String,
    bucket_width: i64,
    max_buckets: i32,
}

impl HistogramCollectorManager {
    /// `new HistogramCollectorManager(field, bucketWidth)`.
    ///
    /// # Errors
    /// As [`with_max_buckets`](Self::with_max_buckets).
    pub fn new(field: impl Into<String>, bucket_width: i64) -> Result<Self> {
        Self::with_max_buckets(field, bucket_width, DEFAULT_MAX_BUCKETS)
    }

    /// `new HistogramCollectorManager(field, bucketWidth, maxBuckets)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a width below 2 or no bucket allowed.
    pub fn with_max_buckets(
        field: impl Into<String>,
        bucket_width: i64,
        max_buckets: i32,
    ) -> Result<Self> {
        if bucket_width < 2 {
            return Err(Error::IllegalArgument(format!(
                "bucketWidth must be at least 2, got: {bucket_width}"
            )));
        }
        if max_buckets < 1 {
            return Err(Error::IllegalArgument(format!(
                "maxBuckets must be at least 1, got: {max_buckets}"
            )));
        }
        Ok(HistogramCollectorManager {
            field: field.into(),
            bucket_width,
            max_buckets,
        })
    }
}

/// `searcher.search(query, histogramCollectorManager)`: each bucket (the
/// value floor-divided by the width) and its count, in bucket order.
///
/// # Errors
/// [`Error::IllegalState`] when the field's doc values are not numeric, or
/// more than `max_buckets` buckets are collected; what the search reports.
pub fn histogram_search(
    searcher: &IndexSearcher<'_, '_>,
    query: &BooleanQuery,
    manager: &HistogramCollectorManager,
) -> Result<BTreeMap<i64, i32>> {
    let bound = Bound {
        manager,
        query,
        point_range: point_range_query(query, &manager.field),
        bulk_collected: Arc::new(Mutex::new(Vec::new())),
    };
    search_manager(searcher, query, &bound)
}

/// The manager with its search's query (Java's `setWeight`).
struct Bound<'m, 'q> {
    manager: &'m HistogramCollectorManager,
    query: &'q BooleanQuery,
    point_range: Option<RangeBounds<'q>>,
    /// `leafBulkCollected`: the segments whose tree was counted.
    bulk_collected: Arc<Mutex<Vec<usize>>>,
}

impl<'a, 'm, 'q> GroupingCollectorManager<'a> for Bound<'m, 'q> {
    type Collector = HistogramCollector<'m, 'q, 'a>;
    type Output = BTreeMap<i64, i32>;

    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(HistogramCollector {
            manager: self.manager,
            query: self.query,
            point_range: self.point_range,
            bulk_collected: self.bulk_collected.clone(),
            counts: FastMap::default(),
            leaf: None,
        })
    }

    /// `reduce(collectors)`: the counts summed, the bucket limit checked
    /// after each.
    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<BTreeMap<i64, i32>> {
        let mut reduced = BTreeMap::new();
        for c in collectors {
            for (bucket, count) in c.counts {
                let slot = reduced.entry(bucket).or_insert(0i32);
                *slot = slot.wrapping_add(count);
                check_max_buckets(reduced.len(), self.manager.max_buckets)?;
            }
        }
        Ok(reduced)
    }
}

/// `HistogramCollector.checkMaxBuckets`.
pub(super) fn check_max_buckets(size: usize, max_buckets: i32) -> Result<()> {
    if i64::try_from(size).unwrap_or(i64::MAX) > i64::from(max_buckets) {
        return Err(Error::IllegalState(format!(
            "Collected {size} buckets, which is more than the configured max number of \
             buckets: {max_buckets}"
        )));
    }
    Ok(())
}

/// A point range's corners: a long range's values, or a packed range's
/// bytes (read with the segment's `bytesToLong`, as Java reads
/// `getLowerPoint()`/`getUpperPoint()`).
#[derive(Clone, Copy)]
enum RangeBounds<'q> {
    Long(i64, i64),
    Packed(&'q [u8], &'q [u8]),
}

impl RangeBounds<'_> {
    /// The inclusive bounds as longs.
    fn resolve(self, to_long: fn(&[u8]) -> i64) -> (i64, i64) {
        match self {
            RangeBounds::Long(min, max) => (min, max),
            RangeBounds::Packed(lower, upper) => (to_long(lower), to_long(upper)),
        }
    }
}

/// `getPointRangeQuery(field)`: the search's query, unwrapped of a lone
/// clause ([`BooleanQuery::lone_clause`], Java's rewrite of a one-clause
/// boolean) and of
/// `unwrapIntoConcreteQuery`'s wrappers (boosts, constant scores, function
/// scores, an `IndexOrDocValuesQuery`'s index side), when it is a point
/// range over `field`.
fn point_range_query<'q>(query: &'q BooleanQuery, field: &str) -> Option<RangeBounds<'q>> {
    let mut clause = query.lone_clause()?;
    loop {
        clause = match clause {
            Clause::ConstantScore(c) => &c.inner,
            Clause::Boost(b) => &b.inner,
            Clause::Boolean(b) => return point_range_query(b, field),
            Clause::PointsRange(p) if p.field == field => {
                return Some(RangeBounds::Long(p.min, p.max));
            }
            Clause::Extended(e) => match &**e {
                ExtendedQuery::FunctionScore(f) => &f.in_query,
                ExtendedQuery::IndexOrDocValues(q) => &q.index_query,
                ExtendedQuery::PointRange(p) if p.field == field => {
                    return Some(RangeBounds::Packed(&p.lower, &p.upper));
                }
                _ => return None,
            },
            _ => return None,
        };
    }
}

/// `HistogramCollector`.
pub struct HistogramCollector<'m, 'q, 'a> {
    manager: &'m HistogramCollectorManager,
    query: &'q BooleanQuery,
    point_range: Option<RangeBounds<'q>>,
    bulk_collected: Arc<Mutex<Vec<usize>>>,
    counts: FastMap<i64, i32>,
    /// The segment's leaf collector; `None` where Java throws
    /// `CollectionTerminatedException`.
    leaf: Option<Leaf<'a>>,
}

/// The leaf collectors.
enum Leaf<'a> {
    /// `HistogramNaiveLeafCollector`: a multi-valued field, each bucket
    /// once per document.
    Naive(Box<dyn SortedNumericDocValues + 'a>),
    /// `HistogramNaiveSingleValuedLeafCollector`.
    NaiveSingle(Box<dyn SortedNumericDocValues + 'a>),
    /// `HistogramLeafCollector`: a single-valued field with a skip index
    /// spanning at most 1024 buckets.
    Skipping(Box<SkippingLeaf<'a>>),
}

impl<'m, 'q, 'a> HistogramCollector<'m, 'q, 'a> {
    /// `getLeafCollector(context)`.
    fn leaf_collector(&mut self, ord: usize, seg: &OpenSegment<'a>) -> Result<Option<Leaf<'a>>> {
        let reader = seg
            .reader
            .ok_or_else(|| Error::MissingSegmentReader("HistogramCollector".into()))?;
        let field = self.manager.field.as_str();
        let Some(info) = reader.field_infos().field_by_name(field) else {
            return Ok(None);
        };
        if info.doc_values_type != DocValuesType::Numeric
            && info.doc_values_type != DocValuesType::SortedNumeric
        {
            return Err(Error::IllegalState(format!(
                "Expected numeric field, but got doc-value type: {}",
                doc_values_type_name(info.doc_values_type)
            )));
        }
        let match_all = matches!(
            leaf_count(seg, self.query)?,
            LeafCount::Whole(n) if i64::try_from(n).ok() == Some(i64::from(reader.max_doc()))
        );
        if match_all || self.point_range.is_some() {
            if let Some(points) = reader.point_values(field)? {
                if can_collect_efficiently(points.as_ref(), self.manager.bucket_width) {
                    let first = {
                        let mut done = self
                            .bulk_collected
                            .lock()
                            .map_err(|_| Error::IllegalState("histogram state poisoned".into()))?;
                        if done.contains(&ord) {
                            false
                        } else {
                            done.push(ord);
                            true
                        }
                    };
                    if first {
                        self.bulk_collect(points.as_ref())?;
                    }
                    return Ok(None);
                }
            }
        }
        let values = get_sorted_numeric(reader, field)?;
        let singleton = match info.doc_values_type {
            DocValuesType::Numeric => true,
            _ => reader.sorted_numeric_is_singleton(field),
        };
        if !singleton {
            return Ok(Some(Leaf::Naive(values)));
        }
        if let Some(skipper) = reader.doc_values_skipper(field)? {
            let width = self.manager.bucket_width;
            let min = skipper.global_min_value().div_euclid(width);
            let max = skipper.global_max_value().div_euclid(width);
            if max.wrapping_sub(min) <= 1024 {
                let len = usize::try_from(max - min + 1).unwrap_or(0);
                return Ok(Some(Leaf::Skipping(Box::new(SkippingLeaf {
                    values,
                    skipper,
                    counts: vec![0; len],
                    leaf_min_bucket: min,
                    up_to_inclusive: -1,
                    up_to_same_bucket: false,
                    up_to_bucket_index: 0,
                }))));
            }
        }
        Ok(Some(Leaf::NaiveSingle(values)))
    }

    /// `PointTreeBulkCollector.collect`.
    fn bulk_collect(&mut self, points: &dyn PointValues) -> Result<()> {
        let to_long = bytes_to_long(points.bytes_per_dimension());
        let mut leaf_min = to_long(points.min_packed_value());
        let mut leaf_max = to_long(points.max_packed_value());
        if let Some(prq) = self.point_range {
            let (min, max) = prq.resolve(to_long);
            leaf_min = leaf_min.max(min);
            leaf_max = leaf_max.min(max);
        }
        let width = self.manager.bucket_width;
        let mut buckets = BucketManager {
            counts: &mut self.counts,
            counter: 0,
            start: leaf_min,
            // The collector's max is exclusive.
            end: bucket_end(leaf_min, width, leaf_max.wrapping_add(1)),
            max: leaf_max.wrapping_add(1),
            non_zero_buckets: 0,
            max_buckets: self.manager.max_buckets,
            width,
            to_long,
            terminated: false,
            error: None,
        };
        points.intersect(&mut buckets)?;
        if let Some(e) = buckets.error.take() {
            return Err(e);
        }
        buckets.finalize_previous(None)
    }
}

/// `bytesToLong(numBytes)`; [`can_collect_efficiently`] admits only 4 and
/// 8 bytes.
pub(super) fn bytes_to_long(num_bytes: i32) -> fn(&[u8]) -> i64 {
    if num_bytes == 4 {
        |b| i64::from(sortable_bytes_to_int(b, 0))
    } else {
        |b| sortable_bytes_to_long(b, 0)
    }
}

/// `min((floorDiv(start, width) + 1) * width, max)`, wrapping as Java's.
pub(super) fn bucket_end(start: i64, width: i64, max: i64) -> i64 {
    start
        .div_euclid(width)
        .wrapping_add(1)
        .wrapping_mul(width)
        .min(max)
}

/// `canCollectEfficiently(pointValues, bucketWidth)`: one dimension, one
/// value per document, 4 or 8 bytes, and at least 512 points per bucket
/// spanned.
fn can_collect_efficiently(points: &dyn PointValues, width: i64) -> bool {
    if points.num_dimensions() != 1 || i64::from(points.doc_count()) != points.size() {
        return false;
    }
    let bytes = points.bytes_per_dimension();
    if bytes != 4 && bytes != 8 {
        return false;
    }
    let to_long = bytes_to_long(bytes);
    let min = to_long(points.min_packed_value()).div_euclid(width);
    let max = to_long(points.max_packed_value()).div_euclid(width);
    points.size() / 512 >= max.wrapping_sub(min)
}

/// `PointTreeBulkCollector.BucketManager` and its intersect visitor; Java's
/// `CollectionTerminatedException` is [`terminated`](Self::terminated),
/// after which every cell is outside.
struct BucketManager<'c> {
    counts: &'c mut FastMap<i64, i32>,
    counter: i32,
    start: i64,
    end: i64,
    max: i64,
    non_zero_buckets: i32,
    max_buckets: i32,
    width: i64,
    to_long: fn(&[u8]) -> i64,
    terminated: bool,
    /// The bucket limit's error, raised inside the visitor.
    error: Option<Error>,
}

impl BucketManager<'_> {
    /// `finalizePreviousBucket(packedValue)`.
    fn finalize_previous(&mut self, packed: Option<&[u8]>) -> Result<()> {
        if self.counter > 0 {
            let slot = self
                .counts
                .entry(self.start.div_euclid(self.width))
                .or_insert(0);
            *slot = slot.wrapping_add(self.counter);
            self.non_zero_buckets += 1;
            self.counter = 0;
            check_max_buckets(
                usize::try_from(self.non_zero_buckets).unwrap_or(usize::MAX),
                self.max_buckets,
            )?;
        }
        if let Some(p) = packed {
            self.start = (self.to_long)(p);
            self.end = bucket_end(self.start, self.width, self.max);
        }
        Ok(())
    }

    fn within_lower(&self, v: &[u8]) -> bool {
        (self.to_long)(v) >= self.start
    }

    fn within_upper(&self, v: &[u8]) -> bool {
        (self.to_long)(v) < self.end
    }

    fn within_range(&self, v: &[u8]) -> bool {
        self.within_lower(v) && self.within_upper(v)
    }

    /// The visitor's preamble: past the bucket, move to `packed`'s; past
    /// the range, stop. `false` once stopped.
    fn advance_to(&mut self, packed: &[u8]) -> bool {
        if self.terminated {
            return false;
        }
        if !self.within_upper(packed) {
            if let Err(e) = self.finalize_previous(Some(packed)) {
                self.error = Some(e);
                self.terminated = true;
                return false;
            }
            if !self.within_upper(packed) {
                self.terminated = true;
                return false;
            }
        }
        true
    }
}

impl IntersectVisitor for BucketManager<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        if !self.advance_to(min_packed) || !self.within_lower(max_packed) {
            return Relation::CellOutsideQuery;
        }
        if self.within_range(min_packed) && self.within_range(max_packed) {
            return Relation::CellInsideQuery;
        }
        Relation::CellCrossesQuery
    }

    /// A cell inside the bucket: Java counts the node's size, one per point
    /// here (one value per document).
    fn visit(&mut self, _doc_id: i32) {
        if !self.terminated {
            self.counter = self.counter.wrapping_add(1);
        }
    }

    fn visit_many(&mut self, doc_ids: &[i32]) {
        if !self.terminated {
            let n = i32::try_from(doc_ids.len()).unwrap_or(i32::MAX);
            self.counter = self.counter.wrapping_add(n);
        }
    }

    fn visit_with_value(&mut self, _doc_id: i32, packed_value: &[u8]) {
        if self.advance_to(packed_value) && self.within_range(packed_value) {
            self.counter = self.counter.wrapping_add(1);
        }
    }
}

/// `HistogramLeafCollector`: counts in a dense per-segment array, a run
/// of documents the skip index puts in one bucket counted without reading
/// their values.
struct SkippingLeaf<'a> {
    values: Box<dyn SortedNumericDocValues + 'a>,
    skipper: DocValuesSkipper<'a>,
    counts: Vec<i32>,
    leaf_min_bucket: i64,
    up_to_inclusive: i32,
    up_to_same_bucket: bool,
    up_to_bucket_index: usize,
}

impl SkippingLeaf<'_> {
    /// `advanceSkipper(doc)`.
    fn advance_skipper(&mut self, doc: i32, width: i64) {
        if doc > self.skipper.max_doc_id(0) {
            self.skipper.advance(doc);
        }
        self.up_to_same_bucket = false;
        if self.skipper.min_doc_id(0) > doc {
            self.up_to_inclusive = self.skipper.min_doc_id(0) - 1;
            return;
        }
        self.up_to_inclusive = self.skipper.max_doc_id(0);
        for level in 0..self.skipper.num_levels() {
            let total = self.skipper.max_doc_id(level) - self.skipper.min_doc_id(level) + 1;
            let min = self.skipper.min_value(level).div_euclid(width);
            let max = self.skipper.max_value(level).div_euclid(width);
            if self.skipper.doc_count(level) == total && min == max {
                self.up_to_inclusive = self.skipper.max_doc_id(level);
                self.up_to_same_bucket = true;
                self.up_to_bucket_index = usize::try_from(min - self.leaf_min_bucket).unwrap_or(0);
            } else {
                break;
            }
        }
    }

    fn collect(&mut self, doc: i32, width: i64) -> Result<()> {
        if doc > self.up_to_inclusive {
            self.advance_skipper(doc, width);
        }
        let index = if self.up_to_same_bucket {
            self.up_to_bucket_index
        } else if self.values.advance_exact(doc)? {
            let bucket = self.values.next_value()?.div_euclid(width);
            usize::try_from(bucket - self.leaf_min_bucket).unwrap_or(usize::MAX)
        } else {
            return Ok(());
        };
        let slot = self
            .counts
            .get_mut(index)
            .ok_or_else(|| Error::IllegalState("value outside the skip index's range".into()))?;
        *slot = slot.wrapping_add(1);
        Ok(())
    }
}

impl<'a> SegmentCollector<'a> for HistogramCollector<'_, '_, 'a> {
    /// `scoreMode()`.
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.finish()?;
        self.leaf = self.leaf_collector(ord, leaf)?;
        Ok(())
    }

    fn collect(&mut self, doc: i32, _score: f32) -> Result<()> {
        let width = self.manager.bucket_width;
        let max_buckets = self.manager.max_buckets;
        match &mut self.leaf {
            None => Ok(()),
            Some(Leaf::Naive(values)) => {
                if values.advance_exact(doc)? {
                    let mut prev = i64::MIN;
                    for _ in 0..values.doc_value_count() {
                        let bucket = values.next_value()?.div_euclid(width);
                        if bucket != prev {
                            let slot = self.counts.entry(bucket).or_insert(0);
                            *slot = slot.wrapping_add(1);
                            check_max_buckets(self.counts.len(), max_buckets)?;
                            prev = bucket;
                        }
                    }
                }
                Ok(())
            }
            Some(Leaf::NaiveSingle(values)) => {
                if values.advance_exact(doc)? {
                    let bucket = values.next_value()?.div_euclid(width);
                    let slot = self.counts.entry(bucket).or_insert(0);
                    *slot = slot.wrapping_add(1);
                    check_max_buckets(self.counts.len(), max_buckets)?;
                }
                Ok(())
            }
            Some(Leaf::Skipping(leaf)) => leaf.collect(doc, width),
        }
    }

    /// `HistogramLeafCollector.finish()`: the segment's counts merged, the
    /// bucket limit checked once.
    fn finish(&mut self) -> Result<()> {
        if let Some(Leaf::Skipping(leaf)) = self.leaf.take() {
            for (i, &c) in leaf.counts.iter().enumerate() {
                if c != 0 {
                    let bucket = leaf.leaf_min_bucket + i64::try_from(i).unwrap_or(0);
                    let slot = self.counts.entry(bucket).or_insert(0);
                    *slot = slot.wrapping_add(c);
                }
            }
            check_max_buckets(self.counts.len(), self.manager.max_buckets)?;
        }
        Ok(())
    }
}
