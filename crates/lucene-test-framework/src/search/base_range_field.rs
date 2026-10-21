//! `BaseRangeFieldQueryTestCase`: random boxes of 1..=4 dimensions --
//! some missing, some sharing a bound or the whole box with an earlier
//! document, some multi-valued -- indexed as range fields and queried by
//! each relation, every hit checked against the relation computed by hand.
//!
//! Java writes through `newIndexWriterConfig` (a random codec, merge policy
//! and buffer) and may `forceMerge(1)`; here the documents are one segment
//! under a plain `IndexWriter`, deleted in one last commit (the same
//! documents: each delete names an id already added), and the
//! `forceMerge` coin is drawn and ignored. Hits are collected unscored, as
//! `FixedBitSetCollector` does.

use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::document::{Document, IndexableField, NumericDocValuesField, Store, StringField};
use lucene_search::collector::ScoreMode;
use lucene_search::query::Clause;
use lucene_util::java_random::JavaRandom;

use super::base_explanation::top_level;
use super::check_hits;
use super::test_index::TestIndex;
use crate::util::{lucene_test_case, test_util};

/// `Range.QueryType`: the relation a query asks of each indexed box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// `INTERSECTS`.
    Intersects,
    /// `WITHIN`.
    Within,
    /// `CONTAINS`.
    Contains,
    /// `CROSSES`.
    Crosses,
}

/// `Range`: a test box, and its relations to another.
pub trait Range: Clone + Debug {
    /// `numDimensions()`.
    fn num_dimensions(&self) -> usize;
    /// `setMin(d, other.getMin(d))`.
    fn copy_min(&mut self, d: usize, other: &Self);
    /// `setMax(d, other.getMax(d))`.
    fn copy_max(&mut self, d: usize, other: &Self);
    /// `isEqual(other)`.
    fn is_equal(&self, other: &Self) -> bool;
    /// `isDisjoint(other)`.
    fn is_disjoint(&self, other: &Self) -> bool;
    /// `isWithin(other)`.
    fn is_within(&self, other: &Self) -> bool;
    /// `contains(other)`.
    fn contains(&self, other: &Self) -> bool;

    /// `relate(other)`: `None` when disjoint.
    fn relate(&self, other: &Self) -> Option<QueryType> {
        if self.is_disjoint(other) {
            None
        } else if self.is_within(other) {
            Some(QueryType::Within)
        } else if self.contains(other) {
            Some(QueryType::Contains)
        } else {
            Some(QueryType::Crosses)
        }
    }
}

/// The `{Int,Long,Float,Double}TestRange` the core subclasses share: a box
/// of one ordered type.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxRange<T> {
    /// Each dimension's minimum.
    pub min: Vec<T>,
    /// Each dimension's maximum.
    pub max: Vec<T>,
}

impl<T: PartialOrd + Copy + Debug> BoxRange<T> {
    /// A box from two draws per dimension, each dimension ordered.
    #[must_use]
    pub fn ordered(a: Vec<T>, b: Vec<T>) -> Self {
        let (min, max) = a
            .into_iter()
            .zip(b)
            .map(|(x, y)| if y < x { (y, x) } else { (x, y) })
            .unzip();
        Self { min, max }
    }
}

impl<T: PartialOrd + Copy + Debug> Range for BoxRange<T> {
    fn num_dimensions(&self) -> usize {
        self.min.len()
    }

    /// Java's `setMin`: a value above the minimum becomes the maximum. A
    /// box of fewer dimensions than asked for ignores `d`, as
    /// `InetAddressRandomRange` (one dimension, whatever was drawn) does.
    fn copy_min(&mut self, d: usize, other: &Self) {
        let d = d.min(self.min.len() - 1);
        let v = other.min[d];
        if self.min[d] < v {
            self.max[d] = v;
        } else {
            self.min[d] = v;
        }
    }

    /// Java's `setMax`: a value below the maximum becomes the minimum.
    fn copy_max(&mut self, d: usize, other: &Self) {
        let d = d.min(self.min.len() - 1);
        let v = other.max[d];
        if self.max[d] > v {
            self.min[d] = v;
        } else {
            self.max[d] = v;
        }
    }

    fn is_equal(&self, other: &Self) -> bool {
        self.min == other.min && self.max == other.max
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        (0..self.min.len()).any(|d| self.min[d] > other.max[d] || self.max[d] < other.min[d])
    }

    fn is_within(&self, other: &Self) -> bool {
        (0..self.min.len()).all(|d| self.min[d] >= other.min[d] && self.max[d] <= other.max[d])
    }

    fn contains(&self, other: &Self) -> bool {
        (0..self.min.len()).all(|d| self.min[d] <= other.min[d] && self.max[d] >= other.max[d])
    }
}

/// A subclass: its range field, its queries, and its random boxes.
pub trait RangeFieldCase {
    /// Its test box.
    type R: Range;
    /// `newRangeField(box)`.
    fn new_range_field(&self, r: &Self::R) -> Box<dyn IndexableField>;
    /// `newIntersectsQuery`/`newContainsQuery`/`newWithinQuery`/`newCrossesQuery`.
    fn new_query(&self, r: &Self::R, query_type: QueryType) -> Clause;
    /// `nextRange(dimensions)`.
    fn next_range(&self, random: &mut JavaRandom, dimensions: usize) -> Self::R;
}

/// One document's boxes; `missing` is Java's `ranges[id][0].isMissing`.
#[derive(Debug, Clone)]
pub struct Doc<R> {
    /// The boxes (one unless multi-valued).
    pub ranges: Vec<R>,
    /// Indexed without a box.
    pub missing: bool,
}

/// `BaseRangeFieldQueryTestCase` over one subclass, drawing from `random`
/// under the given `TEST_NIGHTLY` and multiplier.
pub struct BaseRangeField<'a, C: RangeFieldCase> {
    case: C,
    random: &'a mut JavaRandom,
    multiplier: i32,
}

impl<'a, C: RangeFieldCase> BaseRangeField<'a, C> {
    /// The test case for `case`.
    pub fn new(case: C, random: &'a mut JavaRandom, multiplier: i32) -> Self {
        Self {
            case,
            random,
            multiplier,
        }
    }

    fn at_least(&mut self, i: i32) -> i32 {
        lucene_test_case::at_least(self.random, self.multiplier, i)
    }

    /// `dimension()`: 1 to 4.
    fn dimension(&mut self) -> usize {
        usize::try_from(self.random.next_int_bounded(4) + 1).expect("1..=4")
    }

    /// `testRandomTiny`: ten small indexes, each one leaf.
    pub fn random_tiny(&mut self) {
        for _ in 0..10 {
            self.do_test_random(10, false);
        }
    }

    /// `testRandomMedium`.
    pub fn random_medium(&mut self) {
        self.do_test_random(1000, false);
    }

    /// `testMultiValued`.
    pub fn multi_valued(&mut self) {
        self.do_test_random(1000, true);
    }

    /// `testAllEqual`: every document the same box.
    pub fn all_equal(&mut self) {
        let num_docs = self.at_least(1000);
        let dims = self.dimension();
        let the_range = self.case.next_range(self.random, dims);
        let docs = vec![
            Doc {
                ranges: vec![the_range],
                missing: false,
            };
            usize::try_from(num_docs).expect("positive")
        ];
        self.verify(&docs);
    }

    /// `testLowCardinality`: 2 to 20 distinct boxes.
    pub fn low_cardinality(&mut self) {
        let num_docs = self.at_least(1000);
        let dims = self.dimension();
        let cardinality = test_util::next_int(self.random, 2, 20);
        let diff: Vec<C::R> = (0..cardinality)
            .map(|_| self.case.next_range(self.random, dims))
            .collect();
        let docs: Vec<Doc<C::R>> = (0..num_docs)
            .map(|_| {
                let at = self.random.next_int_bounded(cardinality);
                Doc {
                    ranges: vec![diff[usize::try_from(at).expect("non-negative")].clone()],
                    missing: false,
                }
            })
            .collect();
        self.verify(&docs);
    }

    /// `doTestRandom(count, multiValued)`.
    pub fn do_test_random(&mut self, count: i32, multi_valued: bool) {
        let num_docs = usize::try_from(self.at_least(count)).expect("positive");
        let dims = self.dimension();
        let mut docs: Vec<Doc<C::R>> = Vec::with_capacity(num_docs);
        for id in 0..num_docs {
            let x = usize::try_from(self.random.next_int_bounded(20)).expect("non-negative");
            let mut doc = Doc {
                ranges: vec![self.case.next_range(self.random, dims)],
                missing: false,
            };
            if x == 17 {
                // some docs don't have a box
                doc.missing = true;
                docs.push(doc);
                continue;
            }
            if multi_valued && self.random.next_boolean() {
                // up to 2 boxes
                let n = self.random.next_int_bounded(2) + 1;
                doc.ranges = (0..n)
                    .map(|_| self.case.next_range(self.random, dims))
                    .collect();
            }
            if id > 0 && x < 9 {
                // don't step on missing ranges
                let mut old = None;
                let mut i = 0;
                loop {
                    let old_id = usize::try_from(
                        self.random
                            .next_int_bounded(i32::try_from(id).expect("an int id")),
                    )
                    .expect("non-negative");
                    if !docs[old_id].missing {
                        old = Some(old_id);
                        break;
                    }
                    i += 1;
                    if i > id {
                        break;
                    }
                }
                if let Some(old_id) = old {
                    let old = docs[old_id].ranges[0].clone();
                    let first = &mut doc.ranges[0];
                    if x == dims * 2 {
                        // fully identical box
                        for d in 0..dims {
                            first.copy_min(d, &old);
                            first.copy_max(d, &old);
                        }
                    } else {
                        // Java's `even` is `dimensions % 2`, the same for
                        // every `m`
                        let even = dims % 2;
                        if x < dims * 2 {
                            let d = x / 2;
                            if even == 0 {
                                first.copy_min(d, &old);
                            } else {
                                first.copy_max(d, &old);
                            }
                        }
                    }
                }
            }
            docs.push(doc);
        }
        self.verify(&docs);
    }

    /// `verify(ranges)`: index, then `atLeast(25)` random queries, each hit
    /// (and miss) checked against [`expected_result`].
    ///
    /// # Panics
    /// On the first wrong hit.
    pub fn verify(&mut self, docs: &[Doc<C::R>]) {
        let mut deleted = BTreeSet::new();
        let mut written = Vec::with_capacity(docs.len());
        for (id, d) in docs.iter().enumerate() {
            let mut doc = Document::new();
            doc.add(StringField::new("id", id.to_string(), Store::No));
            doc.add(NumericDocValuesField::new(
                "id",
                i64::try_from(id).expect("small"),
            ));
            if !d.missing {
                for r in &d.ranges {
                    doc.add_boxed(self.case.new_range_field(r));
                }
            }
            written.push(doc);
            if id > 0 && self.random.next_int_bounded(100) == 1 {
                let to_delete = self
                    .random
                    .next_int_bounded(i32::try_from(id).expect("an int id"));
                deleted.insert(usize::try_from(to_delete).expect("non-negative"));
            }
        }
        let _force_merge = self.random.next_boolean();
        let ids: Vec<String> = deleted.iter().map(ToString::to_string).collect();
        let deletes: Vec<(&str, &str)> = ids.iter().map(|id| ("id", id.as_str())).collect();
        let index = TestIndex::build(
            Arc::new(Analyzer::standard(None)),
            &[written],
            &deletes,
            &[],
        );

        let dims = docs[0].ranges[0].num_dimensions();
        let iters = self.at_least(25);
        for iter in 0..iters {
            let query_range = self.case.next_range(self.random, dims);
            let query_type = match self.random.next_int_bounded(4) {
                0 => QueryType::Intersects,
                1 => QueryType::Contains,
                2 => QueryType::Within,
                _ => QueryType::Crosses,
            };
            let q = top_level(self.case.new_query(&query_range, query_type));
            let hits: BTreeSet<usize> = index.with_searcher(|s| {
                check_hits::all_hits(s, &q, ScoreMode::CompleteNoScores)
                    .expect("search")
                    .into_iter()
                    .map(|(doc, _)| usize::try_from(doc).expect("a doc id"))
                    .collect()
            });
            // one segment, deletes in place: doc id == id
            for (id, d) in docs.iter().enumerate() {
                let expected = !deleted.contains(&id)
                    && !d.missing
                    && expected_result(&query_range, &d.ranges, query_type);
                assert_eq!(
                    hits.contains(&id),
                    expected,
                    "FAIL (iter {iter}): id={id} {} \n queryRange={query_range:?}\n boxes={:?}\n queryType={query_type:?}\n deleted?={}",
                    if expected { "should match but did not" } else { "should not match but did" },
                    d.ranges,
                    deleted.contains(&id)
                );
            }
        }
    }
}

/// `expectedResult(queryRange, ranges, queryType)`: any box matches.
pub fn expected_result<R: Range>(query: &R, ranges: &[R], query_type: QueryType) -> bool {
    ranges
        .iter()
        .any(|r| expected_bbox_query_result(query, r, query_type))
}

/// `expectedBBoxQueryResult(queryRange, range, queryType)`.
pub fn expected_bbox_query_result<R: Range>(query: &R, range: &R, query_type: QueryType) -> bool {
    if query.is_equal(range) && query_type != QueryType::Crosses {
        return true;
    }
    let relation = range.relate(query);
    match query_type {
        QueryType::Intersects => relation.is_some(),
        // by definition, ranges that CONTAIN the query also cross it
        QueryType::Crosses => {
            relation == Some(QueryType::Crosses) || relation == Some(QueryType::Contains)
        }
        _ => relation == Some(query_type),
    }
}
