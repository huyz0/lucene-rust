//! Builds a per-segment scorer tree from a [`Clause`]: `Weight.scorerSupplier`
//! for each query kind, and `BooleanWeight.scorerSupplier` +
//! `BooleanScorerSupplier.getInternal` for the boolean composition.

use std::collections::HashMap;

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::{DocInput, PayInput, PosInput, PostingsFlags};
use lucene_util::fixed_bit_set::FixedBitSet;

use super::conjunction::{BlockMaxConjunctionScorer, ConjunctionScorer, LegConjunctionScorer};
use super::disjunction::{Combine, DisjunctionScorer};
use super::leaf::{AllDocs, ConstantScorer, DocList, TermScorer, ZeroScorer};
use super::phrase::{term_positions_cost, PhraseScorer, PhraseTerm};
use super::req::{ReqExclScorer, ReqOptSumScorer};
use super::wand::{cost_with_min_should_match, WandScorer};
use super::{BoxScorer, Mode};
use crate::bulk_scorer::TermLeg;
use crate::field_norms::FieldNorms;
use crate::points_query::PointsInput;
use crate::query::{BooleanQuery, BoostQuery, Clause, PhraseQuery, TermQuery};
use crate::{similarity, sloppy_phrase, GlobalStats, Result};

/// One segment's readers, and the reader-wide statistics to score with.
#[derive(Clone, Copy)]
pub(crate) struct LeafContext<'a> {
    pub(crate) fields: &'a BlockTreeFields,
    pub(crate) doc_in: Option<&'a DocInput<'a>>,
    pub(crate) pos_in: Option<&'a PosInput<'a>>,
    pub(crate) pay_in: Option<&'a PayInput<'a>>,
    pub(crate) live_docs: Option<&'a FixedBitSet>,
    pub(crate) points: Option<&'a PointsInput<'a>>,
    pub(crate) norms: Option<&'a HashMap<String, FieldNorms<'a>>>,
    pub(crate) global: Option<&'a GlobalStats>,
    /// The segment's `maxDoc`, when the caller knows it: a
    /// `MatchAllDocsQuery` matches every document below it. `None` uses the
    /// `max_doc` the clause was built with.
    pub(crate) max_doc: Option<i32>,
    /// The segment's query cache, if any; see `exec::cache`.
    pub(crate) cache: Option<&'a super::cache::SegmentQueryCache>,
    /// The segment's reader, if any: its norms and doc values.
    pub(crate) reader: Option<&'a crate::directory_reader::SegmentReader>,
    /// `IndexSearcher.getSimilarity()` when it is **not** the default BM25:
    /// term and phrase clauses then score through its
    /// [`crate::similarities::SimScorer`] (`TermWeight`/`PhraseWeight`), from
    /// the reader-wide statistics in `global`. `None` -- every caller but
    /// [`crate::multi_segment::search_boolean_query_multi_segment_with_similarity`],
    /// which maps a default BM25 to `None` too -- is the BM25 fast path.
    pub(crate) similarity: Option<&'a dyn crate::similarities::Similarity>,
}

/// `IndexSearcher.collectionStatistics(field)` and
/// `termStatistics(term, ...)` for one term: the reader-wide ones the
/// statistics pass gathered, else this segment's own (a single-segment
/// search, where they are the same).
fn sim_stats(
    ctx: &LeafContext<'_>,
    field: &str,
    term: &[u8],
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    stats: lucene_codecs::blocktree::TermStats,
) -> (
    crate::similarities::CollectionStatistics,
    crate::similarities::TermStatistics,
) {
    let g = match ctx.global.and_then(|g| g.term(field, term)) {
        Some(g) => *g,
        None => crate::CollectionStats {
            doc_freq: i64::from(stats.doc_freq),
            doc_count: i64::from(field_terms.doc_count),
            total_term_freq: stats.total_term_freq,
            max_doc: i64::from(ctx.max_doc.unwrap_or(0)),
            sum_total_term_freq: field_terms.sum_total_term_freq,
            sum_doc_freq: field_terms.sum_doc_freq,
        },
    };
    (g.collection_statistics(), g.term_statistics())
}

/// The scorer for `clause`, or `None` when it matches nothing in this
/// segment (Java's `null` scorer supplier). `boost` is the product of every
/// enclosing `BoostQuery`, folded into the leaves as `createWeight` does.
pub(crate) fn build<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    match term_leg(ctx, clause, boost, mode)? {
        TermForm::Leg(leg) => {
            return Ok(Some(Box::new(TermScorer::new(
                *leg,
                mode == Mode::TopScores,
            ))));
        }
        TermForm::Absent => return Ok(None),
        TermForm::Other => {}
    }
    match clause {
        Clause::Boolean(b) => build_boolean(ctx, b, boost, mode, top_level),
        // `BoostQuery.createWeight`: `query.createWeight(searcher, scoreMode,
        // BoostQuery.this.boost * boost)`, over the rewritten chain.
        Clause::Boost(b) => {
            let (chain, inner) = boost_chain(b);
            build(ctx, inner, chain * boost, mode, top_level)
        }
        // `ConstantScoreQuery`: the inner query runs without scores.
        Clause::ConstantScore(c) => {
            // `ConstantScoreWeight`'s inner weight is `IndexSearcher.createWeight`'s
            // without scores, so `CachingWrapperWeight` caches it: through the
            // segment's query cache here too, whatever the mode around it.
            let Some(inner) = child(ctx, &c.inner, 1.0, Mode::NoScores, false)? else {
                return Ok(None);
            };
            let inner = inner.into_scorer(Mode::NoScores);
            if !mode.needs_scores() {
                return Ok(Some(inner));
            }
            Ok(Some(Box::new(ConstantScorer::new(
                inner,
                c.score * boost,
                mode == Mode::TopScores,
            ))))
        }
        Clause::DisjunctionMax(d) => {
            // Every disjunct a term, scored: `TermDisMaxScorer`, which a
            // disjunction around it can read a block at a time.
            if mode.needs_scores() && d.disjuncts.len() > 1 {
                let mut terms = Vec::with_capacity(d.disjuncts.len());
                let mut all_terms = true;
                for disjunct in &d.disjuncts {
                    match term_leg(ctx, disjunct, boost, mode)? {
                        TermForm::Leg(leg) => {
                            terms.push(TermScorer::new(*leg, mode == Mode::TopScores))
                        }
                        TermForm::Absent => {}
                        TermForm::Other => {
                            all_terms = false;
                            break;
                        }
                    }
                }
                if all_terms {
                    return Ok(match terms.len() {
                        0 => None,
                        1 => terms.pop().map(|t| -> super::BoxScorer<'a> { Box::new(t) }),
                        _ => {
                            let scorer =
                                super::term_dismax::TermDisMaxScorer::new(terms, d.tie_breaker);
                            Some(Box::new(if mode == Mode::TopScores {
                                scorer.with_block_propagator()?
                            } else {
                                scorer
                            }))
                        }
                    });
                }
            }
            let mut subs = Vec::with_capacity(d.disjuncts.len());
            for disjunct in &d.disjuncts {
                if let Some(s) = build(ctx, disjunct, boost, mode, false)? {
                    subs.push(s);
                }
            }
            Ok(match subs.len() {
                0 => None,
                1 => subs.pop(),
                _ => {
                    let scorer = DisjunctionScorer::new(
                        subs,
                        Combine::Max(d.tie_breaker),
                        mode.needs_scores(),
                    );
                    Some(Box::new(if mode == Mode::TopScores {
                        scorer.with_block_propagator()?
                    } else {
                        scorer
                    }))
                }
            })
        }
        Clause::MatchAllDocs(m) => {
            let max_doc = match ctx.max_doc {
                Some(max_doc) => max_doc,
                // A match-all decoded without a maxDoc (the JVM's) needs the
                // segment's; walking to `i32::MAX` would read past its end.
                None if m.max_doc == i32::MAX => return Err(crate::Error::MatchAllWithoutMaxDoc),
                None => m.max_doc,
            };
            if max_doc <= 0 {
                return Ok(None);
            }
            Ok(Some(Box::new(ConstantScorer::new(
                Box::new(AllDocs::new(max_doc)),
                boost,
                mode == Mode::TopScores,
            ))))
        }
        Clause::MatchNoDocs(_) => Ok(None),
        Clause::PointsRange(q) => points_range(ctx, q, boost, mode),
        Clause::Exists(q) => exists(ctx, q, boost, mode),
        Clause::Extended(q) => super::extended::build(ctx, q, boost, mode, top_level),
        // `FuzzyQuery`'s `TopTermsBlendedFreqScoringRewrite`, streamed. The
        // rewrite does not depend on the score mode: an unscored fuzzy
        // clause matches the same reader-wide expansion.
        Clause::Fuzzy(f) => super::extended::fuzzy_sim(ctx, f, boost, mode, top_level),
        Clause::MultiPhrase(m) => super::extended::multi_phrase(ctx, m, boost, mode),
        // `SpanWeight`/`SpanScorer`.
        Clause::Span(s) => super::span::span(ctx, s, boost, mode),
        Clause::Phrase(p) if !p.has_implicit_positions() => {
            super::extended::positional_phrase(ctx, p, boost, mode)
        }
        Clause::Phrase(p) => match phrase(ctx, p, boost, mode)? {
            PhraseForm::Scorer(s) => Ok(Some(s)),
            PhraseForm::Absent => Ok(None),
            PhraseForm::Other => materialized(ctx, clause, boost, mode),
        },
        other => match super::multi_term::multi_term(ctx, other, boost, mode)? {
            Some(scorer) => Ok(scorer),
            None => materialized(ctx, other, boost, mode),
        },
    }
}

/// `PointRangeQuery`'s scorer: a `ConstantScoreWeight` over the documents
/// the BKD walk collects. When every document has a value and the field's
/// own range sits inside the query's, every document matches without a walk
/// (`PointRangeQuery`'s `allDocsMatch`); a dense result is a bitset and a
/// sparse one a sorted list, as `DocIdSetBuilder` switches at `maxDoc / 128`.
fn points_range<'a>(
    ctx: &LeafContext<'a>,
    q: &crate::query::PointsRangeQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let (Some(points), Some(max_doc)) = (ctx.points, ctx.max_doc) else {
        return materialized(ctx, &Clause::PointsRange(q.clone()), boost, mode);
    };
    let Some(field_number) = points.field_number(&q.field) else {
        return Ok(None);
    };
    let Some(field) = points.reader.field(field_number) else {
        return Ok(None);
    };
    let min = crate::points_query::pack_i64(q.min);
    let max = crate::points_query::pack_i64(q.max);
    let top_scores = mode == Mode::TopScores;
    if field.doc_count == max_doc
        && field.num_dims == 1
        && min.as_slice() <= field.min_packed_value.as_slice()
        && max.as_slice() >= field.max_packed_value.as_slice()
    {
        return Ok(Some(Box::new(ConstantScorer::new(
            Box::new(AllDocs::new(max_doc)),
            boost,
            top_scores,
        ))));
    }
    let mut docs = points.reader.range_query(field_number, &min, &max)?;
    if docs.is_empty() {
        return Ok(None);
    }
    let len = usize::try_from(max_doc).unwrap_or(0);
    let inner: BoxScorer<'a> = if docs.len() > len / 128 {
        let mut bits = FixedBitSet::new(len);
        for &d in &docs {
            if let Ok(i) = usize::try_from(d) {
                // FBS: every doc id the walk returns is below `maxDoc`, the
                // set's length; `i < len` holds by construction.
                if i < len {
                    bits.set(i);
                }
            }
        }
        let cardinality = bits.cardinality() as i64;
        Box::new(super::cache::CachedScorer::new(std::sync::Arc::new(
            super::cache::CachedSet::Bits { bits, cardinality },
        )))
    } else {
        lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut docs);
        Box::new(DocList::new(docs, Vec::new()))
    };
    Ok(Some(Box::new(ConstantScorer::new(
        inner, boost, top_scores,
    ))))
}

/// `FieldExistsQuery`'s scorer: a `ConstantScoreWeight` over the documents
/// the field's source has a value for -- every document when the source is
/// dense, the set read off its `IndexedDISI` otherwise, nothing when the
/// segment has no value (Java's `null` iterator).
fn exists<'a>(
    ctx: &LeafContext<'a>,
    q: &crate::query::FieldExistsQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(reader) = ctx.reader else {
        return Err(crate::Error::MissingSegmentReader(q.field.clone()));
    };
    let top_scores = mode == Mode::TopScores;
    let inner: BoxScorer<'a> = match reader.field_exists_docs(&q.field)? {
        crate::directory_reader::ExistsDocs::None => return Ok(None),
        crate::directory_reader::ExistsDocs::All => {
            if reader.max_doc <= 0 {
                return Ok(None);
            }
            Box::new(AllDocs::new(reader.max_doc))
        }
        crate::directory_reader::ExistsDocs::Bits(bits) => {
            let cardinality = bits.cardinality() as i64;
            Box::new(super::cache::CachedScorer::new(std::sync::Arc::new(
                super::cache::CachedSet::Bits { bits, cardinality },
            )))
        }
    };
    Ok(Some(Box::new(ConstantScorer::new(
        inner, boost, top_scores,
    ))))
}

enum PhraseForm<'a> {
    Scorer(BoxScorer<'a>),
    /// A term is not in this segment: the phrase matches nothing.
    Absent,
    /// Not a shape the streaming scorer takes; resolve it up front.
    Other,
}

/// `PhraseWeight.scorer`: a [`PhraseScorer`] over the terms' positions.
fn phrase<'a>(
    ctx: &LeafContext<'a>,
    p: &PhraseQuery,
    boost: f32,
    mode: Mode,
) -> Result<PhraseForm<'a>> {
    if p.terms.len() < 2 {
        // Empty matches nothing and one term is a term query, as
        // `PhraseQuery.rewrite` has it; both are the up-front path's -- but
        // that path scores BM25, so under another similarity the one term
        // runs as the `TermQuery` it rewrites to.
        if let ([only], Some(_)) = (p.terms.as_slice(), ctx.similarity) {
            let t = TermQuery::new(p.field.clone(), only.clone());
            return Ok(match term(ctx, &t, None, boost, mode)? {
                TermForm::Leg(leg) => {
                    PhraseForm::Scorer(Box::new(TermScorer::new(*leg, mode == Mode::TopScores)))
                }
                TermForm::Absent => PhraseForm::Absent,
                TermForm::Other => PhraseForm::Other,
            });
        }
        return Ok(PhraseForm::Other);
    }
    if ctx.similarity.is_some() {
        return sim_phrase(ctx, p, boost, mode);
    }
    let (Some(doc_in), Some(pos_in)) = (ctx.doc_in, ctx.pos_in) else {
        return Ok(PhraseForm::Other);
    };
    let Some(field_terms) = ctx.fields.field(&p.field) else {
        return Ok(PhraseForm::Absent);
    };
    // `BM25Similarity.idfExplain(TermStatistics[])` sums in a double and
    // casts once: an `f32` sum of three or more idfs can be an ulp off.
    let mut idf_sum = 0.0f64;
    let mut match_cost = 0.0f32;
    let mut terms = Vec::with_capacity(p.terms.len());
    for (slot, term) in p.terms.iter().enumerate() {
        let Some(stats) = field_terms.try_seek_exact(term)? else {
            return Ok(PhraseForm::Absent);
        };
        // A pulsed singleton keeps its one posting in the term dictionary,
        // with no `.doc` stream for a lazy cursor to walk.
        if stats.doc_freq <= 1 {
            return Ok(PhraseForm::Other);
        }
        let (df, dc) = match ctx.global.and_then(|g| g.term(&p.field, term)) {
            Some(g) => (g.doc_freq, g.doc_count),
            None => (stats.doc_freq as i64, field_terms.doc_count as i64),
        };
        idf_sum += f64::from(similarity::idf(df, dc));
        match_cost += term_positions_cost(stats.doc_freq as i64, stats.total_term_freq);
        let Some(cursor) = field_terms.lazy_positions(term, doc_in, pos_in)? else {
            return Ok(PhraseForm::Absent);
        };
        terms.push(PhraseTerm {
            cursor,
            slot,
            cost: stats.doc_freq as i64,
        });
    }
    let field_norms = ctx.norms.and_then(|m| m.get(&p.field));
    Ok(PhraseForm::Scorer(Box::new(PhraseScorer::new(
        terms,
        boost * idf_sum as f32,
        p.slop,
        sloppy_phrase::PhraseRepeats::for_phrase(&p.terms),
        field_norms.map(|n| n.cursor()),
        match_cost,
        mode == Mode::TopScores,
        mode.needs_scores(),
    ))))
}

/// [`phrase`] under a similarity other than the default BM25:
/// `PhraseWeight.getStats`' `similarity.scorer(boost, collectionStats,
/// termStats[])` over every term, then the same [`PhraseScorer`] scoring
/// through it. A phrase with a pulsed-singleton term (no `.doc` stream for a
/// lazy cursor) is resolved eagerly ([`crate::phrase_doc_freqs`]) and scored
/// through the same `SimScorer`.
fn sim_phrase<'a>(
    ctx: &LeafContext<'a>,
    p: &PhraseQuery,
    boost: f32,
    mode: Mode,
) -> Result<PhraseForm<'a>> {
    let Some(sim) = ctx.similarity else {
        return Ok(PhraseForm::Other);
    };
    let Some(field_terms) = ctx.fields.field(&p.field) else {
        return Ok(PhraseForm::Absent);
    };
    let mut found = Vec::with_capacity(p.terms.len());
    let mut term_stats = Vec::with_capacity(p.terms.len());
    let mut collection = None;
    for term in &p.terms {
        let Some(stats) = field_terms.try_seek_exact(term)? else {
            return Ok(PhraseForm::Absent);
        };
        let (c, t) = sim_stats(ctx, &p.field, term, field_terms, stats);
        collection.get_or_insert(c);
        term_stats.push(t);
        found.push(stats);
    }
    let Some(collection) = collection else {
        return Ok(PhraseForm::Absent);
    };
    let scorer = sim.scorer(&p.field, boost, &collection, &term_stats);
    let Some(pos_in) = ctx.pos_in else {
        return Err(crate::Error::MissingPosInput);
    };
    let field_norms = ctx.norms.and_then(|m| m.get(&p.field));
    let lazy = ctx.doc_in.filter(|_| found.iter().all(|s| s.doc_freq > 1));
    let Some(doc_in) = lazy else {
        let matched = crate::phrase_doc_freqs(
            field_terms,
            ctx.doc_in,
            pos_in,
            ctx.pay_in,
            ctx.live_docs,
            p,
        )?;
        if matched.is_empty() {
            return Ok(PhraseForm::Absent);
        }
        let mut norms = field_norms.map(|n| n.cursor());
        let mut docs = Vec::with_capacity(matched.len());
        let mut scores = Vec::with_capacity(matched.len());
        for (doc, freq) in matched {
            let norm = match norms.as_mut() {
                Some(n) => n.norm_long(doc)?.unwrap_or(1),
                None => 1,
            };
            docs.push(doc);
            if mode.needs_scores() {
                scores.push(scorer.score(freq, norm));
            }
        }
        return Ok(PhraseForm::Scorer(Box::new(DocList::new(docs, scores))));
    };
    let mut match_cost = 0.0f32;
    let mut terms = Vec::with_capacity(p.terms.len());
    for (slot, (term, stats)) in p.terms.iter().zip(&found).enumerate() {
        match_cost += term_positions_cost(stats.doc_freq as i64, stats.total_term_freq);
        let Some(cursor) = field_terms.lazy_positions(term, doc_in, pos_in)? else {
            return Ok(PhraseForm::Absent);
        };
        terms.push(PhraseTerm {
            cursor,
            slot,
            cost: stats.doc_freq as i64,
        });
    }
    Ok(PhraseForm::Scorer(Box::new(
        PhraseScorer::new(
            terms,
            0.0,
            p.slop,
            sloppy_phrase::PhraseRepeats::for_phrase(&p.terms),
            field_norms.map(|n| n.cursor()),
            match_cost,
            mode == Mode::TopScores,
            mode.needs_scores(),
        )
        .with_sim_scorer(scorer),
    )))
}

/// A chain of nested `BoostQuery`s as `BoostQuery.rewrite` collapses it --
/// `new BoostQuery(in.query, boost * in.boost)`, innermost first, so
/// `b1(b2(b3(q)))` boosts by `b1 * (b2 * b3)` -- and the query under it.
/// The association matters: three factors can round differently the other
/// way, and the product is the BM25 weight's multiplier.
pub(crate) fn boost_chain(b: &BoostQuery) -> (f32, &Clause) {
    // The product innermost first, as `createWeight` multiplies it on the
    // way out: a pass to the innermost clause, then the factors walked back
    // outward -- each step down the chain from the top to one level short
    // of where the last multiplication left off. Chains are a boost or two
    // deep, so the quadratic walk costs nothing a `Vec` would not.
    let mut depth = 1usize;
    let mut inner = b.inner.as_ref();
    while let Clause::Boost(next) = inner {
        depth += 1;
        inner = next.inner.as_ref();
    }
    let factor = |level: usize| -> f32 {
        let mut q = b;
        for _ in 0..level {
            let Clause::Boost(next) = q.inner.as_ref() else {
                unreachable!("level below the chain's depth")
            };
            q = next;
        }
        q.boost
    };
    let mut product = factor(depth - 1);
    for level in (0..depth - 1).rev() {
        product *= factor(level);
    }
    (product, inner)
}

/// What [`term_leg`] made of a clause.
pub(crate) enum TermForm<'a> {
    Leg(Box<TermLeg<'a>>),
    /// The term is not in this segment: the clause matches nothing.
    Absent,
    /// Not a term, a boosted term or a constant-scored term.
    Other,
}

/// A term clause as a [`TermLeg`], with every enclosing boost folded into its
/// weight and a `ConstantScoreQuery` around it as a constant leg -- the form
/// the batch bulk scorers (`BatchScoreBulkScorer`,
/// `BlockMaxConjunctionBulkScorer`, `MaxScoreBulkScorer`) take.
///
/// `TermWeight.scorerSupplier`: a lazy postings cursor, `DocsOnly` when scores
/// are not needed, with `boost * idf` as the BM25 weight.
pub(crate) fn term_leg<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
) -> Result<TermForm<'a>> {
    match clause {
        Clause::Term(t) => term(ctx, t, None, boost, mode),
        Clause::Boost(b) => {
            let (chain, inner) = boost_chain(b);
            match inner {
                Clause::Term(_) | Clause::ConstantScore(_) => {
                    term_leg(ctx, inner, chain * boost, mode)
                }
                _ => Ok(TermForm::Other),
            }
        }
        Clause::ConstantScore(c) => match c.inner.as_ref() {
            Clause::Term(t) => term(ctx, t, Some(c.score * boost), boost, mode),
            _ => Ok(TermForm::Other),
        },
        _ => Ok(TermForm::Other),
    }
}

fn term<'a>(
    ctx: &LeafContext<'a>,
    t: &TermQuery,
    constant: Option<f32>,
    boost: f32,
    mode: Mode,
) -> Result<TermForm<'a>> {
    let Some(doc_in) = ctx.doc_in else {
        return Ok(TermForm::Other);
    };
    let Some(field_terms) = ctx.fields.field(&t.field) else {
        return Ok(TermForm::Absent);
    };
    // The statistics pass's seek, when it made one here (`TermStates`).
    let seeked = match ctx
        .global
        .and_then(|g| g.term_state(&t.field, &t.term, ctx.fields))
    {
        Some(found) => found,
        None => field_terms.seek_term_state(&t.term)?,
    };
    let Some(seeked) = seeked else {
        return Ok(TermForm::Absent);
    };
    let stats = seeked.stats;
    let cost = stats.doc_freq as i64;
    if !mode.needs_scores() || constant.is_some() {
        let cursor = field_terms.lazy_postings_for(&seeked, doc_in, PostingsFlags::DocsOnly)?;
        return Ok(TermForm::Leg(Box::new(match constant {
            Some(score) if mode.needs_scores() => TermLeg::constant(cursor, cost, score),
            _ => TermLeg::filter(cursor, cost),
        })));
    }
    // Impacts only where a threshold can use them (`TermWeight`: `impacts`
    // for `TOP_SCORES`, plain `postings(FREQS)` otherwise).
    let flags = if mode == Mode::TopScores {
        PostingsFlags::Freqs
    } else {
        PostingsFlags::FreqsNoImpacts
    };
    let cursor = field_terms.lazy_postings_for(&seeked, doc_in, flags)?;
    if let Some(sim) = ctx.similarity {
        // `TermWeight`: `similarity.scorer(boost, collectionStats, termStats)`.
        let (collection, term_stats) = sim_stats(ctx, &t.field, &t.term, field_terms, stats);
        let scorer = sim.scorer(&t.field, boost, &collection, &[term_stats]);
        let field_norms = ctx.norms.and_then(|m| m.get(&t.field));
        return Ok(TermForm::Leg(Box::new(TermLeg::scoring_sim(
            cursor,
            scorer,
            field_norms.map(|n| n.cursor()),
            cost,
        ))));
    }
    let (doc_freq, doc_count) = match ctx.global.and_then(|g| g.term(&t.field, &t.term)) {
        Some(g) => (g.doc_freq, g.doc_count),
        None => (cost, field_terms.doc_count as i64),
    };
    let field_norms = ctx.norms.and_then(|m| m.get(&t.field));
    Ok(TermForm::Leg(Box::new(TermLeg::scoring(
        cursor,
        boost * similarity::idf(doc_freq, doc_count),
        field_norms.map(|n| n.cursor()),
        cost,
        (stats.total_term_freq - cost + 1).max(1) as f32,
    ))))
}

/// One clause of a boolean, in the form the bulk scorers can use when it is
/// a term leg, and as a scorer otherwise.
pub(crate) enum Child<'a> {
    Leg(Box<TermLeg<'a>>),
    Scorer(BoxScorer<'a>),
}

impl<'a> Child<'a> {
    pub(crate) fn into_scorer(self, mode: Mode) -> BoxScorer<'a> {
        match self {
            Child::Leg(leg) => Box::new(TermScorer::new(*leg, mode == Mode::TopScores)),
            Child::Scorer(s) => s,
        }
    }

    /// `ScorerSupplier.cost()`.
    pub(crate) fn cost(&self) -> i64 {
        match self {
            Child::Leg(leg) => leg.cost,
            Child::Scorer(s) => s.cost(),
        }
    }

    pub(crate) fn is_leg(&self) -> bool {
        matches!(self, Child::Leg(_))
    }

    pub(crate) fn into_leg(self) -> TermLeg<'a> {
        match self {
            Child::Leg(leg) => *leg,
            Child::Scorer(_) => unreachable!("checked with is_leg"),
        }
    }
}

pub(crate) fn child<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<Child<'a>>> {
    child_led(ctx, clause, boost, mode, top_level, None)
}

/// The `IndexOrDocValuesQuery` a clause is, if it is one.
fn index_or_doc_values_of(
    clause: &Clause,
) -> Option<&crate::extended_query::IndexOrDocValuesQuery> {
    match clause {
        Clause::Extended(e) => match e.as_ref() {
            crate::extended_query::ExtendedQuery::IndexOrDocValues(q) => Some(q),
            _ => None,
        },
        _ => None,
    }
}

/// [`build`], an `IndexOrDocValuesQuery` with the lead cost its boolean
/// hands its `ScorerSupplier.get(leadCost)`.
fn build_led<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
    top_level: bool,
    lead_cost: Option<i64>,
) -> Result<Option<BoxScorer<'a>>> {
    match index_or_doc_values_of(clause) {
        Some(q) if lead_cost.is_some() => {
            super::extended::index_or_doc_values(ctx, q, boost, mode, top_level, lead_cost)
        }
        _ => build(ctx, clause, boost, mode, top_level),
    }
}

/// A boolean's clauses built, as `BooleanWeight.scorerSupplier` gathers
/// their `ScorerSupplier`s and `BooleanScorerSupplier.get(leadCost)` asks
/// each for its scorer, each with the clause it came from. `None` when a
/// required clause has no scorer in this segment.
///
/// Every clause but an `IndexOrDocValuesQuery` is built first; their costs,
/// with each `IndexOrDocValuesQuery`'s index-side cost, give the boolean's
/// `cost()` (`computeCost`: the cheapest required clause, or what
/// `minimum_should_match` optional ones cost) -- the lead cost the
/// `IndexOrDocValuesQuery`s are then built with, choosing points or doc
/// values as `IndexOrDocValuesQuery`'s `get(leadCost)` does. A boolean nested
/// in another uses its own cost (Java passes the smaller of it and its
/// parent's lead cost).
#[allow(clippy::type_complexity)]
pub(crate) fn build_clauses<'a, 'q>(
    ctx: &LeafContext<'a>,
    q: &'q BooleanQuery,
    boost: f32,
    mode: Mode,
    child_top_level: bool,
) -> Result<
    Option<(
        Vec<(Child<'a>, &'q Clause)>,
        Vec<(Child<'a>, &'q Clause)>,
        Vec<(Child<'a>, &'q Clause)>,
        Vec<(Child<'a>, &'q Clause)>,
    )>,
> {
    use super::extended::{index_side, Supplier};
    // (clause, mode, top level, required) per group, in order.
    let groups: [(&'q [Clause], Mode, bool, bool); 4] = [
        (&q.must, mode, child_top_level, true),
        (&q.filter, Mode::NoScores, false, true),
        (&q.should, mode, child_top_level, false),
        (&q.must_not, Mode::NoScores, false, false),
    ];
    let deferred = groups
        .iter()
        .any(|(cs, ..)| cs.iter().any(|c| index_or_doc_values_of(c).is_some()));
    let mut built: [Vec<Option<Child<'a>>>; 4] = Default::default();
    // The index-side cost of each deferred clause, by group and position.
    let mut costs: [Vec<Option<i64>>; 4] = Default::default();
    // `SHOULD` fuzzy clauses flattened into this boolean, by position.
    let mut flattened: Vec<Option<Vec<Child<'a>>>> = (0..q.should.len()).map(|_| None).collect();
    let flatten = mode.needs_scores() && q.minimum_should_match <= 1;
    for (g, (clauses, m, top, required)) in groups.iter().enumerate() {
        for (i, c) in clauses.iter().enumerate() {
            if let (2, true, Clause::Fuzzy(f)) = (g, flatten, c) {
                let children = super::extended::fuzzy_children(ctx, f, boost, *m)?;
                costs[g].push(None);
                built[g].push(None);
                flattened[i] = Some(children);
                continue;
            }
            if deferred && index_or_doc_values_of(c).is_some() {
                let q = index_or_doc_values_of(c).expect("checked");
                match index_side(ctx, &q.index_query)? {
                    Supplier::Absent if *required => return Ok(None),
                    Supplier::Absent => {
                        built[g].push(None);
                        costs[g].push(None);
                    }
                    Supplier::Present(cost) => {
                        built[g].push(None);
                        costs[g].push(Some(cost.unwrap_or(i64::MAX)));
                    }
                }
                continue;
            }
            let child = child(ctx, c, boost, *m, *top)?;
            if child.is_none() && *required {
                return Ok(None);
            }
            costs[g].push(child.as_ref().map(Child::cost));
            built[g].push(child);
        }
    }
    if deferred {
        // `BooleanWeight.scorerSupplier`: exactly `msm` optional clauses
        // present are all required.
        let flat_costs: Vec<i64> = flattened
            .iter()
            .flatten()
            .flatten()
            .map(Child::cost)
            .collect();
        let present_should = costs[2].iter().filter(|c| c.is_some()).count() + flat_costs.len();
        let mut msm = q.minimum_should_match;
        let mut required: Vec<i64> = costs[0]
            .iter()
            .chain(&costs[1])
            .flatten()
            .copied()
            .collect();
        let mut optional: Vec<i64> = costs[2].iter().flatten().copied().collect();
        optional.extend(flat_costs);
        if present_should == msm {
            required.append(&mut optional);
            msm = 0;
        }
        let min_required = required.iter().copied().min();
        let lead = match min_required {
            Some(c) if msm == 0 => c,
            _ => min_required
                .unwrap_or(i64::MAX)
                .min(super::wand::cost_with_min_should_match(&optional, msm)),
        };
        for (g, (clauses, m, top, required)) in groups.iter().enumerate() {
            for (i, c) in clauses.iter().enumerate() {
                if built[g][i].is_some() || costs[g][i].is_none() {
                    continue;
                }
                if index_or_doc_values_of(c).is_none() {
                    continue;
                }
                let child = child_led(ctx, c, boost, *m, *top, Some(lead))?;
                if child.is_none() && *required {
                    return Ok(None);
                }
                built[g][i] = child;
            }
        }
    }
    let [must, filter, should, must_not] = built;
    let pair = |v: Vec<Option<Child<'a>>>, cs: &'q [Clause]| -> Vec<(Child<'a>, &'q Clause)> {
        v.into_iter()
            .zip(cs)
            .filter_map(|(c, clause)| c.map(|c| (c, clause)))
            .collect()
    };
    let mut should_pairs = Vec::with_capacity(should.len());
    for ((c, clause), flat) in should.into_iter().zip(&q.should).zip(flattened) {
        match flat {
            Some(children) => should_pairs.extend(children.into_iter().map(|c| (c, clause))),
            None => should_pairs.extend(c.map(|c| (c, clause))),
        }
    }
    Ok(Some((
        pair(must, &q.must),
        pair(filter, &q.filter),
        should_pairs,
        pair(must_not, &q.must_not),
    )))
}

/// [`child`] with the lead cost an `IndexOrDocValuesQuery` clause is built
/// with (see [`build_clauses`]).
fn child_led<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
    top_level: bool,
    lead_cost: Option<i64>,
) -> Result<Option<Child<'a>>> {
    // `CachingWrapperWeight`: a clause built without scores asks the
    // segment's query cache first.
    if mode == Mode::NoScores {
        if let (Some(cache), Some(max_doc)) = (ctx.cache, ctx.max_doc) {
            // The core's matches, before deletions: no live docs, and no
            // cache for the clauses inside it (Lucene caches those
            // separately, from their own weights).
            let core = LeafContext {
                live_docs: None,
                cache: None,
                ..*ctx
            };
            let cacheable = crate::segment_cacheable::is_cacheable(clause, ctx.reader);
            match cache.scorer_if_cacheable(clause, max_doc, cacheable, || {
                build_led(&core, clause, boost, mode, top_level, lead_cost)
            })? {
                Some(super::cache::CacheResult::Hit(set)) => {
                    return Ok(Some(Child::Scorer(Box::new(
                        super::cache::CachedScorer::new(set),
                    ))))
                }
                Some(super::cache::CacheResult::Empty) => return Ok(None),
                None => {}
            }
        }
    }
    Ok(match term_leg(ctx, clause, boost, mode)? {
        TermForm::Leg(leg) => Some(Child::Leg(leg)),
        TermForm::Absent => None,
        TermForm::Other => {
            build_led(ctx, clause, boost, mode, top_level, lead_cost)?.map(Child::Scorer)
        }
    })
}

/// A clause kind with no streaming scorer here yet, resolved to its matches
/// (and their scores, times `boost`) up front.
fn materialized<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let docs = crate::resolve_clause_docs(
        ctx.fields,
        ctx.doc_in,
        ctx.pos_in,
        ctx.pay_in,
        ctx.live_docs,
        ctx.points,
        clause,
    )?;
    if docs.is_empty() {
        return Ok(None);
    }
    let scores = if mode.needs_scores() {
        let map = crate::clause_scores(
            ctx.fields,
            ctx.doc_in,
            ctx.pos_in,
            ctx.pay_in,
            ctx.live_docs,
            ctx.points,
            clause,
            ctx.norms,
            ctx.global,
        )?;
        docs.iter()
            .map(|d| map.get(d).copied().unwrap_or(0.0) * boost)
            .collect()
    } else {
        Vec::new()
    };
    Ok(Some(Box::new(DocList::new(docs, scores))))
}

/// `BooleanWeight.scorerSupplier` then `BooleanScorerSupplier.get`.
///
/// `top_level` is `setTopLevelScoringClause`: the root of a search, or the
/// only scoring clause of a top-level boolean. It lets a disjunction prune
/// with `WANDScorer` and a conjunction with `BlockMaxConjunctionScorer`.
pub(crate) fn build_boolean<'a>(
    ctx: &LeafContext<'a>,
    q: &BooleanQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    // `setTopLevelScoringClause` passes through to a lone scoring clause.
    let child_top_level = top_level && q.must.len() + q.should.len() == 1;
    let Some((must, filter, should, must_not)) =
        build_clauses(ctx, q, boost, mode, child_top_level)?
    else {
        return Ok(None);
    };
    let strip = |v: Vec<(Child<'a>, &Clause)>| -> Vec<Child<'a>> {
        v.into_iter().map(|(c, _)| c).collect()
    };
    let (must, filter, should, must_not) =
        (strip(must), strip(filter), strip(should), strip(must_not));

    compose(
        must,
        filter,
        should,
        must_not,
        q.minimum_should_match,
        mode,
        top_level,
    )
}

/// The second half of `BooleanWeight.scorerSupplier` and
/// `BooleanScorerSupplier.get`, over clause scorers already built: `must` and
/// `filter` hold only clauses present in this segment (an absent required
/// clause has already made the whole boolean `None`), `should` and
/// `must_not` the present ones among theirs.
pub(crate) fn compose<'a>(
    mut must: Vec<Child<'a>>,
    filter: Vec<Child<'a>>,
    mut should: Vec<Child<'a>>,
    must_not: Vec<Child<'a>>,
    minimum_should_match: usize,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    let mut msm = minimum_should_match;
    // Exactly `msm` optional clauses exist in this segment: all are required.
    if should.len() == msm {
        must.append(&mut should);
        msm = 0;
    }
    if filter.is_empty() && must.is_empty() && should.is_empty() {
        return Ok(None);
    }
    if should.len() < msm {
        return Ok(None);
    }
    if !mode.needs_scores() && msm == 0 && must.len() + filter.len() > 0 {
        should.clear();
    }

    // `BooleanScorerSupplier.cost()`, the lead cost handed to `WANDScorer`.
    let min_required = must.iter().chain(&filter).map(|s| s.cost()).min();
    let lead_cost = match min_required {
        Some(c) if msm == 0 => c,
        _ => {
            let costs: Vec<i64> = should.iter().map(|s| s.cost()).collect();
            min_required
                .unwrap_or(i64::MAX)
                .min(cost_with_min_should_match(&costs, msm))
        }
    };

    let filter_only = should.is_empty() && must.is_empty();
    let should: Vec<BoxScorer<'a>> = should.into_iter().map(|c| c.into_scorer(mode)).collect();
    let must_not: Vec<BoxScorer<'a>> = must_not
        .into_iter()
        .map(|c| c.into_scorer(Mode::NoScores))
        .collect();
    let scorer: BoxScorer<'a> = if should.is_empty() {
        let req = req(filter, must, mode, top_level)?;
        excl(req, must_not)
    } else if filter.is_empty() && must.is_empty() {
        let opt = opt(should, msm, mode, top_level, lead_cost)?;
        excl(opt, must_not)
    } else if msm > 0 {
        let req = excl(req(filter, must, mode, false)?, must_not);
        let opt = opt(should, msm, mode, false, lead_cost)?;
        Box::new(ConjunctionScorer::new(Vec::new(), vec![req, opt]))
    } else {
        debug_assert!(mode.needs_scores());
        let req = excl(req(filter, must, mode, false)?, must_not);
        let opt = opt(should, msm, mode, false, lead_cost)?;
        Box::new(ReqOptSumScorer::new(req, opt, mode == Mode::TopScores)?)
    };
    // `BooleanScorerSupplier.get`: a filter-only boolean collected for top
    // scores is a constant `0`, so it stops once the top hits fill.
    if mode == Mode::TopScores && filter_only {
        return Ok(Some(Box::new(ConstantScorer::new(scorer, 0.0, true))));
    }
    Ok(Some(scorer))
}

/// `BooleanScorerSupplier.getInternal`'s `minShouldMatch > 0` mix: the
/// required side and the optional side (at least `msm` of it must match),
/// which Lucene conjoins with a plain `ConjunctionScorer`. `bulk` runs the
/// pair as a block-max conjunction instead.
pub(crate) fn req_and_opt<'a>(
    must: Vec<Child<'a>>,
    filter: Vec<Child<'a>>,
    should: Vec<Child<'a>>,
    msm: usize,
    mode: Mode,
) -> Result<(BoxScorer<'a>, BoxScorer<'a>)> {
    let min_required = must.iter().chain(&filter).map(|s| s.cost()).min();
    let costs: Vec<i64> = should.iter().map(|s| s.cost()).collect();
    let lead_cost = min_required
        .unwrap_or(i64::MAX)
        .min(cost_with_min_should_match(&costs, msm));
    let should: Vec<BoxScorer<'a>> = should.into_iter().map(|c| c.into_scorer(mode)).collect();
    let req = req(filter, must, mode, false)?;
    let opt = opt(should, msm, mode, false, lead_cost)?;
    Ok((req, opt))
}

/// `BooleanScorerSupplier.req`.
fn req<'a>(
    filter: Vec<Child<'a>>,
    must: Vec<Child<'a>>,
    mode: Mode,
    top_level: bool,
) -> Result<BoxScorer<'a>> {
    if filter.len() + must.len() == 1 {
        if let Some(s) = must.into_iter().next() {
            return Ok(s.into_scorer(mode));
        }
        let s = filter
            .into_iter()
            .next()
            .expect("one filter clause")
            .into_scorer(Mode::NoScores);
        if !mode.needs_scores() {
            return Ok(s);
        }
        return Ok(Box::new(ZeroScorer(s)));
    }
    let block_max = mode == Mode::TopScores && must.len() > 1 && top_level;
    if !block_max && must.iter().chain(&filter).all(Child::is_leg) {
        return Ok(Box::new(LegConjunctionScorer::new(
            filter.into_iter().map(Child::into_leg).collect(),
            must.into_iter().map(Child::into_leg).collect(),
            mode == Mode::TopScores,
        )));
    }
    let filter: Vec<BoxScorer<'a>> = filter
        .into_iter()
        .map(|c| c.into_scorer(Mode::NoScores))
        .collect();
    let mut must: Vec<BoxScorer<'a>> = must.into_iter().map(|c| c.into_scorer(mode)).collect();
    if mode == Mode::TopScores && must.len() > 1 && top_level {
        let block_max: BoxScorer<'a> = Box::new(BlockMaxConjunctionScorer::new(must)?);
        if filter.is_empty() {
            return Ok(block_max);
        }
        must = vec![block_max];
    }
    Ok(Box::new(ConjunctionScorer::new(filter, must)))
}

/// `BooleanScorerSupplier.excl`.
fn excl<'a>(main: BoxScorer<'a>, prohibited: Vec<BoxScorer<'a>>) -> BoxScorer<'a> {
    if prohibited.is_empty() {
        return main;
    }
    let excluded: BoxScorer<'a> = if prohibited.len() == 1 {
        prohibited
            .into_iter()
            .next()
            .expect("one prohibited clause")
    } else {
        Box::new(DisjunctionScorer::new(prohibited, Combine::Sum, false))
    };
    Box::new(ReqExclScorer::new(main, excluded))
}

/// `BooleanScorerSupplier.opt`.
fn opt<'a>(
    mut optional: Vec<BoxScorer<'a>>,
    msm: usize,
    mode: Mode,
    top_level: bool,
    lead_cost: i64,
) -> Result<BoxScorer<'a>> {
    if optional.len() == 1 {
        return Ok(optional.pop().expect("one optional clause"));
    }
    if (mode == Mode::TopScores && top_level) || msm > 1 {
        return Ok(Box::new(WandScorer::new(
            optional,
            msm,
            mode == Mode::TopScores,
            lead_cost,
        )?));
    }
    Ok(Box::new(DisjunctionScorer::new(
        optional,
        Combine::Sum,
        mode.needs_scores(),
    )))
}
