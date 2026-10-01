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
use super::cache::{CacheResult, CachedScorer, CachedSet};
use super::leaf::{ConstantScorer, TermScorer};
use super::{BoxScorer, Mode, Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::TermLeg;
use crate::query::{BooleanQuery, Clause, TermQuery};
use crate::Result;

/// `AbstractMultiTermQueryConstantScoreWrapper.BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD`.
pub(crate) const BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD: usize = 16;

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
    if ctx.doc_in.is_none() {
        return Ok(None);
    }
    let Some((field, terms, _)) = crate::expanded_terms(ctx.fields, clause)? else {
        // The field is not in this segment.
        return Ok(Some(None));
    };
    constant_score_terms(ctx, &field, terms, boost, mode, true).map(Some)
}

/// `AbstractMultiTermQueryConstantScoreWrapper` over one segment's expanded
/// `terms` (term order): up to 16 terms are a constant-scored boolean of
/// them; past that, `blended` (`MultiTermQueryConstantScoreBlendedWrapper`)
/// keeps the 16 highest-`docFreq` terms as iterators and ORs the rest into a
/// bitset, and the plain wrapper (`MultiTermQueryConstantScoreWrapper`, the
/// `CONSTANT_SCORE_REWRITE`) ORs every term into it.
pub(crate) fn constant_score_terms<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
    mut terms: Vec<(Vec<u8>, lucene_codecs::blocktree::SeekedTerm)>,
    boost: f32,
    mode: Mode,
    blended: bool,
) -> Result<Option<BoxScorer<'a>>> {
    let field = field.to_string();
    let Some(doc_in) = ctx.doc_in else {
        return Ok(None);
    };
    if terms.is_empty() {
        return Ok(None);
    }
    let Some(field_terms) = ctx.fields.field(&field) else {
        return Ok(None);
    };
    // `rewriteAsBooleanQuery`: up to 16 terms become `ConstantScoreQuery`
    // around a `BooleanQuery` of `SHOULD` term queries, whose weight
    // `ConstantScoreQuery` creates without scores -- so `IndexSearcher`
    // wraps it in `CachingWrapperWeight`, and a term set used a few times
    // is iterated from the segment's query cache like any other non-scoring
    // clause, whatever the mode of the clause around it. One term rewrites
    // to a `TermQuery`, which is never cached.
    if terms.len() > 1 && terms.len() <= BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD {
        if let (Some(cache), Some(max_doc)) = (ctx.cache, ctx.max_doc) {
            let rewritten = Clause::Boolean(Box::new(BooleanQuery {
                should: terms
                    .iter()
                    .map(|(t, _)| Clause::Term(TermQuery::new(field.clone(), t.clone())))
                    .collect(),
                ..BooleanQuery::default()
            }));
            let inner = match cache.scorer(&rewritten, max_doc, || {
                Ok(Some(term_union(field_terms, doc_in, &terms)?))
            })? {
                // The cached iterator scores the constant itself.
                Some(CacheResult::Hit(set)) => {
                    return Ok(Some(Box::new(CachedScorer::constant(
                        set,
                        boost,
                        mode == Mode::TopScores,
                    ))))
                }
                Some(CacheResult::Empty) => return Ok(None),
                None => term_union(field_terms, doc_in, &terms)?,
            };
            return Ok(Some(Box::new(ConstantScorer::new(
                inner,
                boost,
                mode == Mode::TopScores,
            ))));
        }
    }
    let mut scorers: Vec<BoxScorer<'a>> = Vec::new();
    let max_doc = ctx.max_doc.or(ctx.reader.map(|r| r.max_doc));
    if let (false, Some(max_doc), true) = (
        blended,
        max_doc,
        terms.len() > BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD,
    ) {
        // `MultiTermQueryConstantScoreWrapper`: every term's documents into
        // one `DocIdSetBuilder`.
        let Some(set) = union_set(field_terms, doc_in, &terms, max_doc)? else {
            return Ok(None);
        };
        let inner: BoxScorer<'a> = Box::new(CachedScorer::new(std::sync::Arc::new(set)));
        return Ok(Some(Box::new(ConstantScorer::new(
            inner,
            boost,
            mode == Mode::TopScores,
        ))));
    }
    if terms.len() > BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD {
        let Some(max_doc) = max_doc else {
            return Ok(Some(Box::new(ConstantScorer::new(
                term_union(field_terms, doc_in, &terms)?,
                boost,
                mode == Mode::TopScores,
            ))));
        };
        // Highest `docFreq` first, ties in term order: the first 16 stay
        // iterators, the rest go into one bitset.
        // Only which 16 matters, not the order of the rest: a selection by
        // `(docFreq desc, term order)` on indices, where a full stable sort
        // moved every expanded term (thousands, for a short prefix) around.
        let mut order: Vec<usize> = (0..terms.len()).collect();
        let key = |i: &usize| (std::cmp::Reverse(terms[*i].1.stats.doc_freq), *i);
        order.select_nth_unstable_by_key(BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD, key);
        order[..BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD].sort_unstable_by_key(key);
        let mut keep = vec![false; terms.len()];
        for &i in &order[..BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD] {
            keep[i] = true;
        }
        let mut top = Vec::with_capacity(BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD);
        let mut rest = Vec::with_capacity(terms.len() - BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD);
        let mut slots: Vec<Option<_>> = terms.drain(..).map(Some).collect();
        for &i in &order[..BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD] {
            top.extend(slots[i].take());
        }
        for (i, slot) in slots.into_iter().enumerate() {
            if !keep[i] {
                rest.extend(slot);
            }
        }
        terms = top;
        // `DocIdSetBuilder` over the rest -- see [`union_set`].
        if let Some(set) = union_set(field_terms, doc_in, &rest, max_doc)? {
            scorers.push(Box::new(CachedScorer::new(std::sync::Arc::new(set))));
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
    Ok(Some(Box::new(ConstantScorer::new(
        inner,
        boost,
        mode == Mode::TopScores,
    ))))
}

/// The union of `terms`' postings, documents only.
/// `DocIdSetBuilder` over `terms`' documents in `[0, max_doc)`, or `None`
/// when they have none.
///
/// Java's builder starts sparse -- a growing array of ids -- and upgrades to
/// a bit set only once more than `maxDoc >> 7` ids have been added. Here the
/// terms' summed `docFreq` bounds the ids up front: under that threshold the
/// ids are collected into a sorted, deduplicated list, so a clause whose
/// remaining terms are rare never touches a `maxDoc`-bit set (128 KiB of
/// fresh pages per query on a 1M-document segment, which is what held
/// `mtq_csb` under Lucene); past it, every term's postings are ORed into one
/// bit set a block at a time (`intoBitSet`). Either way the set holds exactly
/// the same documents.
fn union_set<'d>(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    doc_in: &lucene_codecs::postings::DocInput<'d>,
    terms: &[(Vec<u8>, lucene_codecs::blocktree::SeekedTerm)],
    max_doc: i32,
) -> Result<Option<CachedSet>> {
    let len = usize::try_from(max_doc).unwrap_or(0);
    let bound = terms.iter().fold(0usize, |n, (_, t)| {
        n.saturating_add(usize::try_from(t.stats.doc_freq).unwrap_or(usize::MAX))
    });
    if bound <= (len >> 7).max(1) {
        let mut docs = Vec::with_capacity(bound);
        for (_, seeked) in terms {
            let mut cursor =
                field_terms.lazy_postings_for(seeked, doc_in, PostingsFlags::DocsOnly)?;
            let mut doc = cursor.next_doc()?;
            while doc != lucene_codecs::postings::NO_MORE_DOCS {
                if (0..max_doc).contains(&doc) {
                    docs.push(doc);
                }
                doc = cursor.next_doc()?;
            }
        }
        lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut docs);
        return Ok((!docs.is_empty()).then_some(CachedSet::Docs(docs)));
    }
    let mut words = vec![0u64; lucene_util::fixed_bit_set::bits2words(len)];
    let mut reuse = None;
    for (_, seeked) in terms {
        field_terms.or_docs_into(seeked, doc_in, max_doc, &mut words, &mut reuse)?;
    }
    let bits = FixedBitSet::from_words(words, len);
    let cardinality = bits.cardinality() as i64;
    Ok((cardinality > 0).then_some(CachedSet::Bits { bits, cardinality }))
}

fn term_union<'a>(
    field_terms: &'a lucene_codecs::blocktree::FieldTerms,
    doc_in: &'a lucene_codecs::postings::DocInput<'a>,
    terms: &[(Vec<u8>, lucene_codecs::blocktree::SeekedTerm)],
) -> Result<BoxScorer<'a>> {
    let mut legs = Vec::with_capacity(terms.len());
    for (_, seeked) in terms {
        let cursor = field_terms.lazy_postings_for(seeked, doc_in, PostingsFlags::DocsOnly)?;
        legs.push(TermLeg::filter(cursor, seeked.stats.doc_freq as i64));
    }
    Ok(Box::new(TermUnion::new(legs, None)))
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
