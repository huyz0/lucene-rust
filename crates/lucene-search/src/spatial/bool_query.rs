//! The `BooleanQuery`s the BBox and point-vector strategies build, matched
//! (not scored): every one of them ends up under a `ConstantScoreQuery`.

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::ScoringCollector;
use crate::document::geo::{collect_bits, idx, set_doc};
use crate::document::{reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::Result;

/// A `BooleanQuery` (or one clause): what matches, per segment.
#[derive(Debug)]
pub(crate) enum BoolQuery {
    /// A leaf query.
    Clause(Box<dyn DocumentQuery>),
    /// `MUST`, `SHOULD` and `MUST_NOT` clauses, and
    /// `minimumNumberShouldMatch`.
    Bool {
        must: Vec<BoolQuery>,
        should: Vec<BoolQuery>,
        must_not: Vec<BoolQuery>,
        min_should_match: usize,
    },
}

/// A collector keeping a segment's hits as a bitset.
struct Bits(FixedBitSet, Option<i32>);

impl ScoringCollector for Bits {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        set_doc(&mut self.0, doc_id, &mut self.1);
    }
}

impl BoolQuery {
    /// `new BooleanQuery.Builder()` with every clause at `occur`, the
    /// absent ones skipped (`makeQuery(occur, queries...)`).
    pub(crate) fn all(occur: Occur, clauses: Vec<Option<BoolQuery>>) -> BoolQuery {
        let clauses: Vec<BoolQuery> = clauses.into_iter().flatten().collect();
        let (mut must, mut should) = (Vec::new(), Vec::new());
        match occur {
            Occur::Must => must = clauses,
            Occur::Should => should = clauses,
        }
        BoolQuery::Bool {
            must,
            should,
            must_not: Vec::new(),
            min_should_match: 0,
        }
    }

    /// The live documents of `leaf` that match.
    pub(crate) fn matches(&self, leaf: &OpenSegment<'_>, max_doc: usize) -> Result<FixedBitSet> {
        match self {
            BoolQuery::Clause(q) => {
                let mut bits = Bits(FixedBitSet::new(max_doc), None);
                q.score_leaf(leaf, 1.0, &mut bits)?;
                // The clauses collect only documents of this segment.
                debug_assert!(bits.1.is_none(), "a clause collected outside the segment");
                Ok(bits.0)
            }
            BoolQuery::Bool {
                must,
                should,
                must_not,
                min_should_match,
            } => {
                let mut acc: Option<FixedBitSet> = None;
                for m in must {
                    let b = m.matches(leaf, max_doc)?;
                    acc = Some(match acc {
                        None => b,
                        Some(mut a) => {
                            a.and(&b);
                            a
                        }
                    });
                }
                // SHOULD clauses: required when there is no MUST, or by
                // `minimumNumberShouldMatch`.
                let required = if must.is_empty() {
                    (*min_should_match).max(1)
                } else {
                    *min_should_match
                };
                if required > 0 {
                    if should.is_empty() {
                        // a pure negative (or empty) query matches nothing
                        return Ok(FixedBitSet::new(max_doc));
                    }
                    let mut counts = vec![0usize; max_doc];
                    for s in should {
                        s.matches(leaf, max_doc)?.for_each_set_bit(|d| {
                            if let Some(c) = counts.get_mut(d) {
                                *c += 1;
                            }
                        });
                    }
                    let mut b = FixedBitSet::new(max_doc);
                    let mut bad = None;
                    for (d, &c) in counts.iter().enumerate() {
                        if c >= required {
                            set_doc(&mut b, i32::try_from(d).unwrap_or(-1), &mut bad);
                        }
                    }
                    acc = Some(match acc {
                        None => b,
                        Some(mut a) => {
                            a.and(&b);
                            a
                        }
                    });
                }
                let mut acc = acc.unwrap_or_else(|| FixedBitSet::new(max_doc));
                for n in must_not {
                    acc.and_not(&n.matches(leaf, max_doc)?);
                }
                Ok(acc)
            }
        }
    }
}

/// `BooleanClause.Occur`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Occur {
    Must,
    Should,
}

/// `ConstantScoreQuery(booleanQuery)`: every match at the boost.
#[derive(Debug)]
pub(crate) struct ConstantScoreBool(pub(crate) BoolQuery);

impl DocumentQuery for ConstantScoreBool {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let max_doc = idx(reader(leaf)?.max_doc);
        let bits = self.0.matches(leaf, max_doc)?;
        collect_bits(leaf, &bits, boost, collector);
        Ok(())
    }
}

/// The set bits of a segment bitset as doc ids, ascending.
pub(crate) fn docs_of(bits: &FixedBitSet) -> Vec<i32> {
    let mut docs = Vec::with_capacity(bits.cardinality());
    bits.for_each_set_bit(|d| docs.extend(i32::try_from(d).ok()));
    docs
}
