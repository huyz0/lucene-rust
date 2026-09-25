//! The multi-term family as constant-score scorers: `TermInSetQuery` and
//! the `MultiTermQuery`s (`PrefixQuery`, `WildcardQuery`, `RegexpQuery`)
//! under their default `CONSTANT_SCORE_BLENDED_REWRITE`.
//!
//! `AbstractMultiTermQueryConstantScoreWrapper`: a clause expanding to at
//! most 16 terms (`BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD`) is a
//! constant-scored disjunction of those terms' postings; past that,
//! `MultiTermQueryConstantScoreBlendedWrapper` keeps the 16 terms with the
//! highest `docFreq` as iterators and ORs every other term's documents into
//! one bitset, and the disjunction runs over both. Iterating lazily is what
//! lets a top-hits search stop once its constant-scored slots fill; the
//! bitset is what keeps a clause with thousands of rare terms from carrying
//! thousands of iterators.

use lucene_codecs::postings::PostingsFlags;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::build::LeafContext;
use super::cache::{CachedScorer, CachedSet};
use super::leaf::{ConstantScorer, TermScorer};
use super::{BoxScorer, Mode, Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::TermLeg;
use crate::query::Clause;
use crate::{blocktree, Result};

/// `AbstractMultiTermQueryConstantScoreWrapper.BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD`.
const BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD: usize = 16;

/// The scorer for a multi-term `clause`: `Ok(None)` when it is not one this
/// takes (the caller resolves it up front), `Ok(Some(None))` when it matches
/// nothing in this segment.
pub(crate) fn multi_term<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
) -> Result<Option<Option<BoxScorer<'a>>>> {
    if !matches!(
        clause,
        Clause::TermInSet(_) | Clause::Prefix(_) | Clause::Wildcard(_) | Clause::Regexp(_)
    ) {
        return Ok(None);
    }
    let Some(doc_in) = ctx.doc_in else {
        return Ok(None);
    };
    let Some((field, mut terms, _)) = crate::expanded_terms(ctx.fields, clause)? else {
        // The field is not in this segment.
        return Ok(Some(None));
    };
    if terms.is_empty() {
        return Ok(Some(None));
    }
    let Some(field_terms) = ctx.fields.field(&field) else {
        return Ok(Some(None));
    };
    let pe = |e| -> crate::Error { blocktree::Error::Postings(e).into() };
    let mut scorers: Vec<BoxScorer<'a>> = Vec::new();
    if terms.len() > BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD {
        let Some(max_doc) = ctx.max_doc else {
            return Ok(None);
        };
        // Highest `docFreq` first, ties in term order: the first 16 stay
        // iterators, the rest go into one bitset.
        terms.sort_by_key(|t| std::cmp::Reverse(t.1.stats.doc_freq));
        let rest = terms.split_off(BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD);
        let len = usize::try_from(max_doc).unwrap_or(0);
        let mut bits = FixedBitSet::new(len);
        let mut cardinality = 0i64;
        for (_, seeked) in &rest {
            let mut cursor =
                field_terms.lazy_postings_for(seeked, doc_in, PostingsFlags::DocsOnly)?;
            let mut doc = cursor.next_doc().map_err(pe)?;
            while doc != NO_MORE_DOCS {
                if let Ok(i) = usize::try_from(doc) {
                    // FBS: `i < len`, the set's length, checked here.
                    if i < len && !bits.get(i) {
                        bits.set(i);
                        cardinality += 1;
                    }
                }
                doc = cursor.next_doc().map_err(pe)?;
            }
        }
        if cardinality > 0 {
            scorers.push(Box::new(CachedScorer::new(std::sync::Arc::new(
                CachedSet::Bits { bits, cardinality },
            ))));
        }
    }
    let mut legs = Vec::with_capacity(terms.len());
    for (_, seeked) in &terms {
        let cursor = field_terms.lazy_postings_for(seeked, doc_in, PostingsFlags::DocsOnly)?;
        legs.push(TermLeg::filter(cursor, seeked.stats.doc_freq as i64));
    }
    let inner: BoxScorer<'a> = match (scorers.pop(), legs.len()) {
        (None, 1) => Box::new(TermScorer::new(legs.pop().expect("one leg"), false)),
        (bits, _) => Box::new(TermUnion::new(legs, bits)),
    };
    // `ConstantScoreQuery`'s score: the boost.
    Ok(Some(Some(Box::new(ConstantScorer::new(
        inner,
        boost,
        mode == Mode::TopScores,
    )))))
}

/// `DisjunctionDISIApproximation` over at most 16 term iterators and the
/// blended rewrite's bitset: the union's documents, matches only. The terms
/// are held directly and scanned for the next document, where a generic
/// disjunction goes through a heap and a virtual call per clause per move --
/// three times the cost per document on a `terms` clause under a `MUST`.
pub(crate) struct TermUnion<'a> {
    legs: Vec<TermLeg<'a>>,
    /// The blended rewrite's other terms, as one bitset.
    bits: Option<BoxScorer<'a>>,
    doc: i32,
    cost: i64,
}

impl<'a> TermUnion<'a> {
    pub(crate) fn new(legs: Vec<TermLeg<'a>>, bits: Option<BoxScorer<'a>>) -> Self {
        let cost = legs
            .iter()
            .map(|l| l.cost)
            .chain(bits.as_ref().map(|b| b.cost()))
            .fold(0i64, i64::saturating_add);
        Self {
            legs,
            bits,
            doc: -1,
            cost,
        }
    }

    fn min_doc(&self) -> i32 {
        let legs = self.legs.iter().map(TermLeg::doc_id);
        let bits = self.bits.as_ref().map(|b| b.doc_id());
        legs.chain(bits).min().unwrap_or(NO_MORE_DOCS)
    }
}

impl Scorer for TermUnion<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        let cur = self.doc;
        for leg in self.legs.iter_mut() {
            if leg.doc_id() <= cur {
                leg.next_doc()?;
            }
        }
        if let Some(b) = self.bits.as_mut() {
            if b.doc_id() <= cur {
                b.next_doc()?;
            }
        }
        self.doc = self.min_doc();
        Ok(self.doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        for leg in self.legs.iter_mut() {
            if leg.doc_id() < target {
                leg.advance(target)?;
            }
        }
        if let Some(b) = self.bits.as_mut() {
            if b.doc_id() < target {
                b.advance(target)?;
            }
        }
        self.doc = self.min_doc();
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.cost
    }

    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }

    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
}
