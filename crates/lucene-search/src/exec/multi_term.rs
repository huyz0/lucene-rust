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

use lucene_codecs::blocktree::SeekedTerm;
use lucene_codecs::postings::PostingsFlags;
use lucene_util::fixed_bit_set::FixedBitSet;

use super::build::LeafContext;
use super::cache::{CacheResult, CachedScorer, CachedSet};
use super::leaf::{ConstantScorer, TermScorer};
use super::{BoxScorer, Mode, Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::TermLeg;
use crate::java_priority_queue::JavaPriorityQueue;
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
    // The prefix, wildcard and regexp clauses stream their terms, as
    // `MultiTermQueryConstantScoreBlendedWrapper` does off its `TermsEnum`;
    // a term set is already a list.
    let streamed = match clause {
        Clause::Prefix(q) => ctx.fields.field(&q.field).map(|ft| {
            let pattern = lucene_codecs::wildcard::WildcardPattern::prefix(&q.prefix);
            blended_stream(ctx, &q.field, ft.intersect_states(&pattern), boost, mode)
        }),
        Clause::Wildcard(q) => ctx.fields.field(&q.field).map(|ft| {
            let pattern = lucene_codecs::wildcard::WildcardPattern::new(&q.pattern);
            blended_stream(ctx, &q.field, ft.intersect_states(&pattern), boost, mode)
        }),
        Clause::Regexp(q) => match ctx.fields.field(&q.field) {
            Some(ft) => {
                let pattern = lucene_codecs::regexp::RegexpPattern::new(q.pattern.as_bytes())?;
                Some(blended_stream(
                    ctx,
                    &q.field,
                    ft.regexp_intersect_states(&pattern),
                    boost,
                    mode,
                ))
            }
            None => None,
        },
        _ => {
            let Some((field, terms, _)) = crate::expanded_terms(ctx.fields, clause)? else {
                // The field is not in this segment.
                return Ok(Some(None));
            };
            let mut stream = StreamedTerms::new(ctx, &field, true);
            for (term, seeked) in terms {
                stream.push(term, seeked)?;
                if stream.settled() {
                    break;
                }
            }
            return stream.finish(ctx, &field, boost, mode).map(Some);
        }
    };
    // `None`: the field is not in this segment.
    streamed.transpose().map(Option::flatten).map(Some)
}

/// [`constant_score_terms`] (blended) over a clause's `terms` as its
/// `TermsEnum` yields them; see [`StreamedTerms`].
fn blended_stream<'a, I>(
    ctx: &LeafContext<'a>,
    field: &str,
    terms: I,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>>
where
    I: Iterator<Item = lucene_codecs::blocktree::Result<(Vec<u8>, SeekedTerm)>>,
{
    let mut stream = StreamedTerms::new(ctx, field, true);
    for t in terms {
        let (term, seeked) = t?;
        stream.push(term, seeked)?;
        if stream.settled() {
            break;
        }
    }
    stream.finish(ctx, field, boost, mode)
}

/// [`constant_score_terms`] fed one term at a time, in term order, without
/// holding them all, as the wrappers consume their `TermsEnum`. A short
/// prefix expands to tens of thousands of terms; collecting them first (term
/// bytes, metadata, then a selection over the lot) was a fifth of
/// `mtq_csb`'s time.
///
/// - Up to 16 terms take [`constant_score_terms`]' boolean rewrite
///   (`collectTerms`, `rewriteAsBooleanQuery`), as does a segment without the
///   inputs the union needs.
/// - Past that, `MultiTermQueryConstantScoreBlendedWrapper` adds a term with
///   `docFreq <= 512` (`POSTINGS_PRE_PROCESS_THRESHOLD`) straight into its
///   `DocIdSetBuilder` and offers the others to Lucene's 16-entry priority
///   queue by `docFreq` (`insertWithOverflow`: an incoming tie replaces the
///   heap top), whatever the queue drops going into the set as well. The
///   split never changes a match, but it sets the clause's cost, and so the
///   order a disjunction partitions its clauses in.
///   `MultiTermQueryConstantScoreWrapper` adds every term to the set.
/// - A term whose `docFreq` is the field's `docCount` matches every document
///   the others can, so both wrappers drop the rest and run that term alone
///   ([`Self::settled`] tells a caller it can stop walking the terms).
///
/// None of this decides which documents match, only what it costs: the
/// score is a constant.
pub(crate) struct StreamedTerms<'a> {
    blended: bool,
    /// The terms so far, while there are at most 16 (or all of them, when
    /// the union cannot be built here).
    head: Vec<(Vec<u8>, SeekedTerm)>,
    /// The field and `.doc` input every term's postings come from, and the
    /// segment's `maxDoc`: `None` when the segment lacks one.
    inputs: Option<Inputs<'a>>,
    /// Past 16 terms: the union and the kept terms.
    stream: Option<Stream<'a>>,
    /// The field's `docCount`, when the field is in the segment.
    field_doc_count: Option<i32>,
    /// A term matching every document with the field, once one is pushed.
    dense: Option<(Vec<u8>, SeekedTerm)>,
}

/// `MultiTermQueryConstantScoreBlendedWrapper.POSTINGS_PRE_PROCESS_THRESHOLD`:
/// a term this rare goes into the set without being offered to the queue.
const POSTINGS_PRE_PROCESS_THRESHOLD: i32 = 512;

type Inputs<'a> = (
    &'a lucene_codecs::blocktree::FieldTerms,
    &'a lucene_codecs::postings::DocInput<'a>,
    i32,
);

/// `highFrequencyTerms`: the kept terms as `(docFreq, term)`, by `docFreq`.
type KeptTerms<T> = JavaPriorityQueue<(i32, T), fn(&(i32, T), &(i32, T)) -> bool>;

/// `highFrequencyTerms`' `lessThan`: `a.cost() < b.cost()`.
fn kept_terms<T>() -> KeptTerms<T> {
    JavaPriorityQueue::new(BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD, |a, b| a.0 < b.0)
}

struct Stream<'a> {
    union: UnionBuilder<'a>,
    /// The kept terms; blended only.
    top: KeptTerms<SeekedTerm>,
}

impl<'a> StreamedTerms<'a> {
    /// Ready for `field`'s terms in `ctx`'s segment; `blended` picks the
    /// blended rewrite over the plain constant-score one.
    pub(crate) fn new(ctx: &LeafContext<'a>, field: &str, blended: bool) -> Self {
        let max_doc = ctx.max_doc.or(ctx.reader.map(|r| r.max_doc));
        let inputs = match (ctx.fields.field(field), ctx.doc_in, max_doc) {
            (Some(ft), Some(doc_in), Some(max_doc)) => Some((ft, doc_in, max_doc)),
            _ => None,
        };
        Self {
            blended,
            head: Vec::with_capacity(BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD + 1),
            inputs,
            stream: None,
            field_doc_count: ctx.fields.field(field).map(|ft| ft.doc_count),
            dense: None,
        }
    }

    /// Whether a term already pushed decides the clause, so the rest need
    /// not be walked (`if (fieldDocCount == docFreq)`).
    pub(crate) fn settled(&self) -> bool {
        self.dense.is_some()
    }

    /// The next term, in term order.
    pub(crate) fn push(&mut self, term: Vec<u8>, seeked: SeekedTerm) -> Result<()> {
        if self.dense.is_some() {
            return Ok(());
        }
        if self.field_doc_count == Some(seeked.stats.doc_freq) {
            self.dense = Some((term, seeked));
            return Ok(());
        }
        let Some((ft, doc_in, max_doc)) = self.inputs else {
            self.head.push((term, seeked));
            return Ok(());
        };
        let Some(stream) = self.stream.as_mut() else {
            self.head.push((term, seeked));
            if self.head.len() > BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD {
                // Past the boolean rewrite: replay the head into a stream.
                let mut stream = Stream {
                    union: UnionBuilder::new(max_doc, ft),
                    top: kept_terms(),
                };
                for (_, seeked) in std::mem::take(&mut self.head) {
                    stream.add(self.blended, ft, doc_in, seeked)?;
                }
                self.stream = Some(stream);
            }
            return Ok(());
        };
        stream.add(self.blended, ft, doc_in, seeked)
    }

    /// The clause's scorer over every term pushed.
    pub(crate) fn finish(
        self,
        ctx: &LeafContext<'a>,
        field: &str,
        boost: f32,
        mode: Mode,
    ) -> Result<Option<BoxScorer<'a>>> {
        if let Some(dense) = self.dense {
            // `new ConstantScoreQuery(new TermQuery(term))`.
            return constant_score_terms(ctx, field, vec![dense], boost, mode);
        }
        let (Some(stream), Some((ft, doc_in, _))) = (self.stream, self.inputs) else {
            return constant_score_terms(ctx, field, self.head, boost, mode);
        };
        // `DisjunctionDISIApproximation` orders its clauses by cost, the
        // costliest first (a stable sort over the queue's heap order).
        let mut top = stream.top.into_heap_order();
        top.sort_by_key(|t| std::cmp::Reverse(t.0));
        let mut legs = Vec::with_capacity(top.len());
        for (_, seeked) in &top {
            let cursor = ft.lazy_postings_for(seeked, doc_in, PostingsFlags::DocsOnly)?;
            legs.push(TermLeg::filter(cursor, seeked.stats.doc_freq as i64));
        }
        let bits: Option<BoxScorer<'a>> = stream.union.build().map(|(set, cost)| {
            let scorer = CachedScorer::new(std::sync::Arc::new(set));
            Box::new(match cost {
                Some(cost) => scorer.with_cost(cost),
                None => scorer,
            }) as BoxScorer<'a>
        });
        let inner: BoxScorer<'a> = match (bits, legs.is_empty()) {
            (None, true) => return Ok(None),
            // `MultiTermQueryConstantScoreWrapper`: the set alone.
            (Some(bits), true) => bits,
            (bits, false) => Box::new(TermUnion::new(legs, bits)),
        };
        Ok(Some(Box::new(ConstantScorer::new(
            inner,
            boost,
            mode == Mode::TopScores,
        ))))
    }
}

impl<'a> Stream<'a> {
    /// The next term: kept as an iterator if it is past
    /// [`POSTINGS_PRE_PROCESS_THRESHOLD`] and the queue takes it (blended),
    /// else into the union -- along with whichever kept term it displaces.
    fn add(
        &mut self,
        blended: bool,
        ft: &'a lucene_codecs::blocktree::FieldTerms,
        doc_in: &'a lucene_codecs::postings::DocInput<'a>,
        seeked: SeekedTerm,
    ) -> Result<()> {
        if !blended || seeked.stats.doc_freq <= POSTINGS_PRE_PROCESS_THRESHOLD {
            return self.union.add(ft, doc_in, &seeked);
        }
        match self
            .top
            .insert_with_overflow_dropped((seeked.stats.doc_freq, seeked))
        {
            Some((_, out)) => self.union.add(ft, doc_in, &out),
            None => Ok(()),
        }
    }
}

/// `AbstractMultiTermQueryConstantScoreWrapper` over one segment's expanded
/// `terms` (term order) that [`StreamedTerms`] did not stream: up to 16
/// terms are a constant-scored boolean of them (`rewriteAsBooleanQuery`),
/// and more, on a segment with no `maxDoc` for the union's set, a
/// constant-scored disjunction of all of them.
pub(crate) fn constant_score_terms<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
    terms: Vec<(Vec<u8>, lucene_codecs::blocktree::SeekedTerm)>,
    boost: f32,
    mode: Mode,
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
    // No `maxDoc` to size a set by (only a caller without a reader passes
    // none; [`StreamedTerms`] builds the union whenever it can): a
    // constant-scored disjunction of every term.
    Ok(Some(Box::new(ConstantScorer::new(
        term_union(field_terms, doc_in, &terms)?,
        boost,
        mode == Mode::TopScores,
    ))))
}

/// `DocIdSetBuilder`: the documents in `[0, max_doc)` of the terms added to
/// it, built into a [`CachedSet`].
///
/// Java's builder starts sparse -- a growing array of ids -- and upgrades to
/// a bit set once more than `maxDoc >> 7` ids would be held. Here a term's
/// `docFreq` says up front whether adding it crosses that line: below it the
/// ids are collected into a list, sorted and deduplicated at the end, so a
/// clause whose terms are rare never touches a `maxDoc`-bit set (128 KiB of
/// fresh pages per query on a 1M-document segment, which is what held
/// `mtq_csb` under Lucene); past it, every term's postings are ORed into one
/// bit set a block at a time (`intoBitSet`), one [`crate::bit_set_pool`]
/// hands back once the set is dropped. Either way the set holds exactly the
/// same documents.
struct UnionBuilder<'d> {
    max_doc: i32,
    /// `maxDoc >>> 7`: the most ids the sparse buffers may allocate before
    /// the set upgrades.
    threshold: usize,
    /// `totalAllocated`, and the last buffer's capacity and length, as
    /// `DocIdSetBuilder` grows its buffers: when they would pass
    /// [`Self::threshold`] decides between a list and a bit set, and so the
    /// cost the disjunction is ordered by.
    total_allocated: usize,
    last_buffer: Option<(usize, usize)>,
    /// `counter`: every term's `docFreq`, once a bit set.
    counter: i64,
    /// `numValuesPerDoc`: the field's `sumDocFreq / docCount`.
    values_per_doc: f64,
    docs: Vec<i32>,
    words: Option<Vec<u64>>,
    /// One cursor reset for every term ORed into `words`.
    reuse: Option<lucene_codecs::postings::LazyDocsCursor<'d>>,
}

impl<'d> UnionBuilder<'d> {
    fn new(max_doc: i32, field_terms: &lucene_codecs::blocktree::FieldTerms) -> Self {
        let len = usize::try_from(max_doc).unwrap_or(0);
        let (doc_count, value_count) = (field_terms.doc_count, field_terms.sum_doc_freq);
        Self {
            max_doc,
            threshold: len >> 7,
            total_allocated: 0,
            last_buffer: None,
            counter: 0,
            values_per_doc: if doc_count <= 0 || value_count < 0 {
                1.0
            } else {
                value_count as f64 / f64::from(doc_count)
            },
            docs: Vec::new(),
            words: None,
            reuse: None,
        }
    }

    fn len(&self) -> usize {
        usize::try_from(self.max_doc).unwrap_or(0)
    }

    fn add(
        &mut self,
        field_terms: &lucene_codecs::blocktree::FieldTerms,
        doc_in: &lucene_codecs::postings::DocInput<'d>,
        term: &SeekedTerm,
    ) -> Result<()> {
        let df = usize::try_from(term.stats.doc_freq).unwrap_or(usize::MAX);
        if self.words.is_none() && !self.grow_buffers(df) {
            self.counter = self.docs.len() as i64;
            // A cleared set this thread has spare, or a fresh one: the
            // `DocIdSetBuilder` allocation Lucene makes per query on a warm
            // heap.
            let len = self.len();
            let mut words = crate::bit_set_pool::take(len).map_or_else(
                || vec![0u64; lucene_util::fixed_bit_set::bits2words(len)],
                FixedBitSet::into_words,
            );
            for &doc in &self.docs {
                // ARITH: every listed id is in `[0, max_doc)`.
                let i = doc as u32 as usize;
                if let Some(w) = words.get_mut(i >> 6) {
                    *w |= 1u64 << (i & 63);
                }
            }
            self.docs = Vec::new();
            self.words = Some(words);
        }
        if let Some(words) = self.words.as_mut() {
            self.counter = self.counter.saturating_add(i64::from(term.stats.doc_freq));
            return Ok(field_terms.or_docs_into(
                term,
                doc_in,
                self.max_doc,
                words,
                &mut self.reuse,
            )?);
        }
        let mut cursor = field_terms.lazy_postings_for(term, doc_in, PostingsFlags::DocsOnly)?;
        let mut doc = cursor.next_doc()?;
        while doc != lucene_codecs::postings::NO_MORE_DOCS {
            if (0..self.max_doc).contains(&doc) {
                self.docs.push(doc);
            }
            doc = cursor.next_doc()?;
        }
        Ok(())
    }

    /// `DocIdSetBuilder.grow(numDocs)` while still on buffers: whether
    /// `num_docs` more fit under the threshold (`ensureBufferCapacity`), or
    /// the builder must upgrade to a bit set.
    fn grow_buffers(&mut self, num_docs: usize) -> bool {
        if self.total_allocated.saturating_add(num_docs) > self.threshold {
            return false;
        }
        // `additionalCapacity`: what has been allocated so far, at least
        // `numDocs + 1` and 32, never past the threshold.
        let additional = |total: usize| {
            num_docs
                .saturating_add(1)
                .max(total)
                .max(32)
                .min(self.threshold - total)
        };
        let (cap, len) = match self.last_buffer {
            None => {
                let add = additional(self.total_allocated);
                self.total_allocated += add;
                (add, 0)
            }
            Some((cap, len)) if cap - len >= num_docs => (cap, len),
            Some((cap, len)) if len < cap - (cap >> 3) => {
                let add = additional(self.total_allocated);
                self.total_allocated += add;
                (cap + add, len)
            }
            Some(_) => {
                let add = additional(self.total_allocated);
                self.total_allocated += add;
                (add, 0)
            }
        };
        self.last_buffer = Some((cap, len.saturating_add(num_docs)));
        true
    }

    /// The set, or `None` when no document was added, with the cost
    /// `DocIdSetBuilder.build` gives a bit set: `round(counter /
    /// numValuesPerDoc)`. A list costs its length.
    fn build(self) -> Option<(CachedSet, Option<i64>)> {
        let len = self.len();
        match self.words {
            Some(words) => {
                let bits = FixedBitSet::from_words(words, len);
                let cardinality = bits.cardinality() as i64;
                let cost = (self.counter as f64 / self.values_per_doc + 0.5).floor() as i64;
                (cardinality > 0).then_some((CachedSet::Bits { bits, cardinality }, Some(cost)))
            }
            None => {
                let mut docs = self.docs;
                lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut docs);
                (!docs.is_empty()).then_some((CachedSet::Docs(docs), None))
            }
        }
    }
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
    // One term needs no disjunction around it.
    if legs.len() == 1 {
        let leg = legs.pop().expect("one leg");
        return Ok(Box::new(TermScorer::new(leg, false)));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The queue is Lucene's: it keeps 16 terms of the highest `docFreq`s,
    /// an incoming tie replacing the heap top, and hands back every other
    /// term once.
    #[test]
    fn the_queue_keeps_sixteen_terms_as_lucene_does() {
        // Few distinct `docFreq`s, so ties are everywhere, in a scrambled order.
        let dfs: Vec<i32> = (0..200u32)
            .map(|i| ((i * 37 + 11) % 23) as i32 * 100)
            .collect();
        let mut top = kept_terms();
        let mut out = Vec::new();
        for (at, &df) in dfs.iter().enumerate() {
            out.extend(top.insert_with_overflow_dropped((df, at)).map(|t| t.1));
        }
        let kept = top.into_heap_order();
        assert_eq!(kept.len(), BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD);
        let mut by_df: Vec<i32> = dfs.clone();
        by_df.sort_unstable_by(|a, b| b.cmp(a));
        let mut kept_dfs: Vec<i32> = kept.iter().map(|t| t.0).collect();
        kept_dfs.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(kept_dfs, by_df[..BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD]);
        // An incoming tie replaces the heap top, which goes to the union.
        let mut tied = kept_terms();
        for at in 0..BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD {
            assert!(tied.insert_with_overflow_dropped((7, at)).is_none());
        }
        assert_eq!(tied.insert_with_overflow_dropped((7, 99)), Some((7, 0)));
        assert_eq!(tied.insert_with_overflow_dropped((6, 98)), Some((6, 98)));
        let mut every: Vec<usize> = out;
        every.extend(kept.iter().map(|t| t.1));
        every.sort_unstable();
        assert_eq!(every, (0..dfs.len()).collect::<Vec<_>>(), "each term once");
    }
}
