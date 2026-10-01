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
///   `DocIdSetBuilder` and offers the others to a 16-entry priority queue by
///   `docFreq` (`insertWithOverflow`: an earlier term keeps its place on a
///   tie), whatever the queue drops going into the set as well.
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
    /// Terms seen so far.
    seen: usize,
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

struct Stream<'a> {
    union: UnionBuilder<'a>,
    /// The kept terms as `(docFreq, position, term)`; blended only.
    top: Vec<(i32, usize, SeekedTerm)>,
    /// The position in `top` of its lowest-ranked term.
    lowest: usize,
}

/// Whether the term at `(docFreq, position)` `a` ranks above `b`: a higher
/// `docFreq`, then an earlier position.
fn ranks_above(a: (i32, usize), b: (i32, usize)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
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
            seen: 0,
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
        let at = self.seen;
        self.seen = self.seen.saturating_add(1);
        let Some((ft, doc_in, max_doc)) = self.inputs else {
            self.head.push((term, seeked));
            return Ok(());
        };
        let Some(stream) = self.stream.as_mut() else {
            self.head.push((term, seeked));
            if self.head.len() > BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD {
                // Past the boolean rewrite: replay the head into a stream.
                let mut stream = Stream {
                    union: UnionBuilder::new(max_doc),
                    top: Vec::with_capacity(BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD),
                    lowest: 0,
                };
                for (i, (_, seeked)) in std::mem::take(&mut self.head).into_iter().enumerate() {
                    stream.add(self.blended, ft, doc_in, i, seeked)?;
                }
                self.stream = Some(stream);
            }
            return Ok(());
        };
        stream.add(self.blended, ft, doc_in, at, seeked)
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
        let (Some(mut stream), Some((ft, doc_in, _))) = (self.stream, self.inputs) else {
            return constant_score_terms(ctx, field, self.head, boost, mode);
        };
        stream
            .top
            .sort_unstable_by(|a, b| match ranks_above((a.0, a.1), (b.0, b.1)) {
                true => std::cmp::Ordering::Less,
                false => std::cmp::Ordering::Greater,
            });
        let mut legs = Vec::with_capacity(stream.top.len());
        for (_, _, seeked) in &stream.top {
            let cursor = ft.lazy_postings_for(seeked, doc_in, PostingsFlags::DocsOnly)?;
            legs.push(TermLeg::filter(cursor, seeked.stats.doc_freq as i64));
        }
        let bits: Option<BoxScorer<'a>> = stream
            .union
            .build()
            .map(|set| Box::new(CachedScorer::new(std::sync::Arc::new(set))) as BoxScorer<'a>);
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
    /// Term `at` (its position in term order): kept as an iterator if it is
    /// past [`POSTINGS_PRE_PROCESS_THRESHOLD`] and among the 16
    /// highest-ranked so far (blended), else into the union -- along with
    /// whichever kept term it displaces.
    fn add(
        &mut self,
        blended: bool,
        ft: &'a lucene_codecs::blocktree::FieldTerms,
        doc_in: &'a lucene_codecs::postings::DocInput<'a>,
        at: usize,
        seeked: SeekedTerm,
    ) -> Result<()> {
        if !blended || seeked.stats.doc_freq <= POSTINGS_PRE_PROCESS_THRESHOLD {
            return self.union.add(ft, doc_in, &seeked);
        }
        let key = (seeked.stats.doc_freq, at);
        match offer(&mut self.top, &mut self.lowest, key, seeked) {
            Some(out) => self.union.add(ft, doc_in, &out),
            None => Ok(()),
        }
    }
}

/// `PriorityQueue.insertWithOverflow` over the 16 kept terms, as
/// `(docFreq, position, term)`: `item` is kept while there is room, or in
/// place of the lowest-ranked kept term if it ranks above it. Returns
/// whichever term is not kept -- `item` itself, or the one it displaced.
/// `lowest` tracks the position of the lowest-ranked kept term.
fn offer<T>(
    top: &mut Vec<(i32, usize, T)>,
    lowest: &mut usize,
    key: (i32, usize),
    item: T,
) -> Option<T> {
    if top.len() < BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD {
        top.push((key.0, key.1, item));
        *lowest = lowest_rank(top);
        return None;
    }
    let low = &top[*lowest];
    if !ranks_above(key, (low.0, low.1)) {
        return Some(item);
    }
    let evicted = std::mem::replace(&mut top[*lowest], (key.0, key.1, item));
    *lowest = lowest_rank(top);
    Some(evicted.2)
}

/// The position in `top` of its lowest-ranked term.
fn lowest_rank<T>(top: &[(i32, usize, T)]) -> usize {
    let mut low = 0;
    for (i, t) in top.iter().enumerate().skip(1) {
        if ranks_above((top[low].0, top[low].1), (t.0, t.1)) {
            low = i;
        }
    }
    low
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
    /// The most ids the sparse list holds before the set upgrades.
    threshold: usize,
    docs: Vec<i32>,
    words: Option<Vec<u64>>,
    /// One cursor reset for every term ORed into `words`.
    reuse: Option<lucene_codecs::postings::LazyDocsCursor<'d>>,
}

impl<'d> UnionBuilder<'d> {
    fn new(max_doc: i32) -> Self {
        let len = usize::try_from(max_doc).unwrap_or(0);
        Self {
            max_doc,
            threshold: (len >> 7).max(1),
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
        if self.words.is_none() && self.docs.len().saturating_add(df) > self.threshold {
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

    /// The set, or `None` when no document was added.
    fn build(self) -> Option<CachedSet> {
        let len = self.len();
        match self.words {
            Some(words) => {
                let bits = FixedBitSet::from_words(words, len);
                let cardinality = bits.cardinality() as i64;
                (cardinality > 0).then_some(CachedSet::Bits { bits, cardinality })
            }
            None => {
                let mut docs = self.docs;
                lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut docs);
                (!docs.is_empty()).then_some(CachedSet::Docs(docs))
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

    /// The queue keeps exactly the 16 highest `(docFreq desc, position asc)`
    /// keys of any sequence -- ties included, where the earlier term stays
    /// -- and hands back every other key once.
    #[test]
    fn the_queue_keeps_the_sixteen_highest_ranked_terms() {
        // Few distinct `docFreq`s, so ties are everywhere, in a scrambled order.
        let dfs: Vec<i32> = (0..200u32)
            .map(|i| ((i * 37 + 11) % 23) as i32 * 100)
            .collect();
        let mut top = Vec::new();
        let mut lowest = 0;
        let mut out = Vec::new();
        for (at, &df) in dfs.iter().enumerate() {
            out.extend(offer(&mut top, &mut lowest, (df, at), at));
        }
        let mut kept: Vec<usize> = top.iter().map(|t| t.2).collect();
        kept.sort_unstable();
        let mut want: Vec<usize> = (0..dfs.len()).collect();
        want.sort_by_key(|&i| (std::cmp::Reverse(dfs[i]), i));
        want.truncate(BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD);
        want.sort_unstable();
        assert_eq!(kept, want);
        out.sort_unstable();
        let rest: Vec<usize> = (0..dfs.len()).filter(|i| !want.contains(i)).collect();
        assert_eq!(out, rest, "every term not kept goes to the union, once");
        assert!(ranks_above((5, 9), (4, 0)));
        assert!(ranks_above((5, 1), (5, 2)), "a tie keeps the earlier term");
        assert!(!ranks_above((5, 2), (5, 2)));
    }
}
