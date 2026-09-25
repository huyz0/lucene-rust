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
use super::req::{ReqExclScorer, ReqOptSumScorer};
use super::wand::{cost_with_min_should_match, WandScorer};
use super::{BoxScorer, Mode};
use crate::bulk_scorer::TermLeg;
use crate::field_norms::FieldNorms;
use crate::points_query::PointsInput;
use crate::query::{BooleanQuery, BoostQuery, Clause, TermQuery};
use crate::{similarity, GlobalStats, Result};

/// One segment's readers, and the reader-wide statistics to score with.
pub(crate) struct LeafContext<'a> {
    pub(crate) fields: &'a BlockTreeFields,
    pub(crate) doc_in: Option<&'a DocInput<'a>>,
    pub(crate) pos_in: Option<&'a PosInput<'a>>,
    pub(crate) pay_in: Option<&'a PayInput<'a>>,
    pub(crate) live_docs: Option<&'a FixedBitSet>,
    pub(crate) points: Option<&'a PointsInput<'a>>,
    pub(crate) norms: Option<&'a HashMap<String, FieldNorms<'a>>>,
    pub(crate) global: Option<&'a GlobalStats>,
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
            let Some(inner) = build(ctx, &c.inner, 1.0, Mode::NoScores, false)? else {
                return Ok(None);
            };
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
            let mut subs = Vec::with_capacity(d.disjuncts.len());
            for disjunct in &d.disjuncts {
                if let Some(s) = build(ctx, disjunct, boost, mode, false)? {
                    subs.push(s);
                }
            }
            Ok(match subs.len() {
                0 => None,
                1 => subs.pop(),
                _ => Some(Box::new(DisjunctionScorer::new(
                    subs,
                    Combine::Max(d.tie_breaker),
                    mode.needs_scores(),
                ))),
            })
        }
        Clause::MatchAllDocs(m) => {
            if m.max_doc <= 0 {
                return Ok(None);
            }
            Ok(Some(Box::new(ConstantScorer::new(
                Box::new(AllDocs::new(m.max_doc)),
                boost,
                mode == Mode::TopScores,
            ))))
        }
        Clause::MatchNoDocs(_) => Ok(None),
        other => materialized(ctx, other, boost, mode),
    }
}

/// A chain of nested `BoostQuery`s as `BoostQuery.rewrite` collapses it --
/// `new BoostQuery(in.query, boost * in.boost)`, innermost first, so
/// `b1(b2(b3(q)))` boosts by `b1 * (b2 * b3)` -- and the query under it.
/// The association matters: three factors can round differently the other
/// way, and the product is the BM25 weight's multiplier.
pub(crate) fn boost_chain(b: &BoostQuery) -> (f32, &Clause) {
    let mut factors = vec![b.boost];
    let mut inner = b.inner.as_ref();
    while let Clause::Boost(next) = inner {
        factors.push(next.boost);
        inner = next.inner.as_ref();
    }
    let mut product = factors.pop().expect("at least one boost");
    while let Some(outer) = factors.pop() {
        product *= outer;
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
    let Some(seeked) = field_terms.seek_term_state(&t.term)? else {
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
    let cursor = field_terms.lazy_postings_for(&seeked, doc_in, PostingsFlags::Freqs)?;
    let (doc_freq, doc_count) = match ctx.global.and_then(|g| g.term(&t.field, &t.term)) {
        Some(g) => (g.doc_freq, g.doc_count),
        None => (cost, field_terms.doc_count as i64),
    };
    let field_norms = ctx.norms.and_then(|m| m.get(&t.field));
    Ok(TermForm::Leg(Box::new(TermLeg::scoring(
        cursor,
        boost * similarity::idf(doc_freq, doc_count),
        field_norms.map(|n| n.cursor()),
        field_norms.map_or(similarity::UNNORMED_FIELD_LENGTH, |n| n.avg_field_length),
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
    Ok(match term_leg(ctx, clause, boost, mode)? {
        TermForm::Leg(leg) => Some(Child::Leg(leg)),
        TermForm::Absent => None,
        TermForm::Other => build(ctx, clause, boost, mode, top_level)?.map(Child::Scorer),
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
    let mut must = Vec::with_capacity(q.must.len());
    for c in &q.must {
        match child(ctx, c, boost, mode, child_top_level)? {
            Some(s) => must.push(s),
            None => return Ok(None),
        }
    }
    let mut filter = Vec::with_capacity(q.filter.len());
    for c in &q.filter {
        match child(ctx, c, boost, Mode::NoScores, false)? {
            Some(s) => filter.push(s),
            None => return Ok(None),
        }
    }
    let mut should = Vec::with_capacity(q.should.len());
    for c in &q.should {
        if let Some(s) = child(ctx, c, boost, mode, child_top_level)? {
            should.push(s);
        }
    }
    let mut must_not = Vec::with_capacity(q.must_not.len());
    for c in &q.must_not {
        if let Some(s) = child(ctx, c, boost, Mode::NoScores, false)? {
            must_not.push(s);
        }
    }

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
