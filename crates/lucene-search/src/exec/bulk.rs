//! Bulk scorers and their selection: `Weight.bulkScorer` for the top-level
//! query, and `BooleanScorerSupplier.booleanScorer()`'s choice between
//! `BatchScoreBulkScorer` (one term), `BlockMaxConjunctionBulkScorer` (a
//! conjunction), `MaxScoreBulkScorer` (a disjunction, optionally filtered),
//! `ReqExclBulkScorer` around any of them for `MUST_NOT` clauses, and
//! `DefaultBulkScorer` over the scorer tree for every other shape.
//!
//! The term-shaped bulk scorers are the batch ports in `bulk_scorer`; they
//! take their clauses as [`TermLeg`]s, which a term, a boosted term and a
//! constant-scored term all become (`build::term_leg`). `MaxScoreBulkScorer`
//! and `BlockMaxConjunctionBulkScorer` also run over clauses that are not
//! all terms, as in Lucene ([`ScorerLeg`], `Bulk::ScorerConjunction`). Any
//! other shape runs on the scorer tree, as Lucene does for the shapes it has
//! no bulk scorer for -- except two where Lucene's tree never skips a
//! document and ours need not either: a dismax of terms (`DisMaxBulk`) and
//! `MUST` + `SHOULD` with a minimum (the same two scorers, as a block-max
//! conjunction).

use lucene_util::fixed_bit_set::FixedBitSet;

use super::build::{boost_chain, child, compose, term_leg, Child, LeafContext, TermForm};
use super::conjunction::ConjunctionScorer;
use super::disjunction::{Combine, DisjunctionScorer};
use super::leaf::ZeroScorer;
use super::{BoxScorer, Mode, NO_MORE_DOCS};
use crate::bulk_scorer::{
    min_competitive_score, score_term_window, BulkLeg, ConjunctionBulk, DisMaxBulk, DocScores,
    MaxScore, MaxScoreLeg, ReqOptBulk, TermLeg,
};
use crate::collector::ScoringCollector;
use crate::query::{BooleanQuery, Clause};
use crate::Result;

/// A bulk scorer for one segment.
pub(crate) enum Bulk<'a> {
    /// `DefaultBulkScorer` over a scorer tree, with the threshold last handed
    /// to it.
    Scorer(BoxScorer<'a>, f32),
    /// `BatchScoreBulkScorer`, with its batch buffer.
    Term(Box<TermLeg<'a>>, DocScores),
    /// `BlockMaxConjunctionBulkScorer`; the legs cheapest first.
    Conjunction(Vec<TermLeg<'a>>, ConjunctionBulk),
    /// `BlockMaxConjunctionBulkScorer` over clauses that are not all terms
    /// (a nested boolean, a dismax); cheapest first.
    ScorerConjunction(Vec<BoxScorer<'a>>, ConjunctionBulk),
    /// `MaxScoreBulkScorer`, with `filteredOptionalBulkScorer`'s filter.
    Disjunction(Vec<TermLeg<'a>>, Option<BoxScorer<'a>>, MaxScore),
    /// `MaxScoreBulkScorer` over clauses that are not all terms (a nested
    /// boolean, a dismax), with `filteredOptionalBulkScorer`'s filter.
    ScorerDisjunction(Vec<ScorerLeg<'a>>, Option<BoxScorer<'a>>, MaxScore),
    /// `ReqOptSumScorer`'s shape, a batch at a time: the required legs
    /// (cheapest first), then the optional ones. See `ReqOptBulk`.
    ReqOpt(Vec<TermLeg<'a>>, Vec<TermLeg<'a>>, Box<ReqOptBulk>),
    /// `ReqExclBulkScorer`.
    ReqExcl(Box<Bulk<'a>>, BoxScorer<'a>),
    /// A dismax of terms, a window at a time. See `DisMaxBulk`.
    DisMax(Vec<TermLeg<'a>>, Box<DisMaxBulk>),
}

impl<'a> Bulk<'a> {
    pub(crate) fn scorer(s: BoxScorer<'a>) -> Self {
        Bulk::Scorer(s, 0.0)
    }

    /// Which bulk scorer this is, for tests that pin the dispatch.
    #[cfg(test)]
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Bulk::Scorer(..) => "scorer",
            Bulk::Term(..) => "term",
            Bulk::Conjunction(..) => "conjunction",
            Bulk::ScorerConjunction(..) => "scorer_conjunction",
            Bulk::Disjunction(_, None, _) => "disjunction",
            Bulk::Disjunction(_, Some(_), _) => "filtered_disjunction",
            Bulk::ScorerDisjunction(..) => "scorer_disjunction",
            Bulk::ReqOpt(..) => "req_opt",
            Bulk::ReqExcl(..) => "req_excl",
            Bulk::DisMax(..) => "dismax",
        }
    }

    /// `BulkScorer.score(collector, acceptDocs, min, max)`: collects the
    /// matches in `[min, max)` and returns the next document to score, at or
    /// past `max` ([`NO_MORE_DOCS`] when there is none).
    pub(crate) fn score<C: ScoringCollector + ?Sized>(
        &mut self,
        mode: Mode,
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        match self {
            Bulk::Scorer(s, published) => {
                default_score(&mut **s, published, mode, live_docs, collector, min, max)
            }
            Bulk::Term(leg, buf) => score_term_window(leg, buf, live_docs, collector, min, max),
            Bulk::Conjunction(legs, state) => state.score(legs, live_docs, collector, min, max),
            Bulk::ScorerConjunction(scorers, state) => {
                state.score(scorers, live_docs, collector, min, max)
            }
            Bulk::Disjunction(legs, filter, state) => {
                let filter = filter.as_deref_mut().map(|f| f as &mut dyn super::Scorer);
                state.score(legs, filter, live_docs, collector, min, max)
            }
            Bulk::ScorerDisjunction(legs, filter, state) => {
                let filter = filter.as_deref_mut().map(|f| f as &mut dyn super::Scorer);
                state.score(legs, filter, live_docs, collector, min, max)
            }
            Bulk::ReqOpt(req, opt, state) => state.score(req, opt, live_docs, collector, min, max),
            Bulk::DisMax(legs, state) => state.score(legs, live_docs, collector, min, max),
            Bulk::ReqExcl(req, excl) => {
                req_excl_score(req, &mut **excl, mode, live_docs, collector, min, max)
            }
        }
    }
}

/// `DefaultBulkScorer.score`, with `TopScoreDocCollector`'s leaf collector's
/// half of the pruning handshake: `setScorer` and every competitive hit pass
/// a newly published threshold down as `setMinCompetitiveScore`.
fn default_score<C: ScoringCollector + ?Sized>(
    scorer: &mut dyn super::Scorer,
    published: &mut f32,
    mode: Mode,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
    min: i32,
    max: i32,
) -> Result<i32> {
    let prune = mode == Mode::TopScores;
    let needs_scores = mode.needs_scores();
    let two_phase = scorer.two_phase();
    if prune {
        publish(scorer, published, collector)?;
    }
    let mut doc = scorer.doc_id();
    if doc < min {
        doc = if doc == min - 1 {
            scorer.next_doc()?
        } else {
            scorer.advance(min)?
        };
    }
    while doc < max {
        if live_docs.is_none_or(|l| l.get_doc(doc)) && (!two_phase || scorer.matches()?) {
            let score = if needs_scores { scorer.score()? } else { 0.0 };
            collector.collect(doc, score);
            if prune {
                publish(scorer, published, collector)?;
            }
        }
        doc = scorer.next_doc()?;
    }
    Ok(doc)
}

#[inline]
fn publish<C: ScoringCollector + ?Sized>(
    scorer: &mut dyn super::Scorer,
    published: &mut f32,
    collector: &C,
) -> Result<()> {
    let m = min_competitive_score(collector);
    if m > *published {
        scorer.set_min_competitive_score(m)?;
        *published = m;
    }
    Ok(())
}

/// `ReqExclBulkScorer.score`: the required bulk scorer, run only over the
/// ranges between excluded documents -- and past a whole run of consecutive
/// excluded documents at once (`docIDRunEnd`), which on a dense excluded
/// term skips entire postings blocks of the required clause.
fn req_excl_score<C: ScoringCollector + ?Sized>(
    req: &mut Bulk<'_>,
    excl: &mut dyn super::Scorer,
    mode: Mode,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
    min: i32,
    max: i32,
) -> Result<i32> {
    let mut up_to = min;
    let mut excl_doc = excl.doc_id();
    while up_to < max {
        if excl_doc < up_to {
            excl_doc = excl.advance(up_to)?;
        }
        if excl_doc == up_to {
            if !excl.two_phase() {
                up_to = excl.doc_id_run_end().min(max);
            } else if excl.matches()? {
                up_to += 1;
            }
            excl_doc = excl.next_doc()?;
        } else {
            up_to = req.score(mode, live_docs, collector, up_to, excl_doc.min(max))?;
        }
    }
    if up_to == max {
        up_to = req.score(mode, live_docs, collector, up_to, up_to)?;
    }
    Ok(up_to)
}

/// `Weight.bulkScorer` for a top-level clause.
pub(crate) fn bulk_clause<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
) -> Result<Option<Bulk<'a>>> {
    if let Clause::Boolean(b) = clause {
        return bulk_boolean(ctx, b, boost, mode);
    }
    if let Clause::Boost(b) = clause {
        let (chain, inner) = boost_chain(b);
        if let Clause::Boolean(inner) = inner {
            return bulk_boolean(ctx, inner, chain * boost, mode);
        }
    }
    let (dismax_boost, inner) = match clause {
        Clause::Boost(b) => {
            let (chain, inner) = boost_chain(b);
            (chain * boost, inner)
        }
        other => (boost, other),
    };
    if let Clause::DisjunctionMax(d) = inner {
        if let Some(bulk) = bulk_dismax(ctx, d, dismax_boost, mode)? {
            return Ok(bulk);
        }
    }
    Ok(match child(ctx, clause, boost, mode, true)? {
        Some(Child::Leg(leg)) => Some(Bulk::Term(leg, DocScores::default())),
        Some(Child::Scorer(s)) => Some(Bulk::scorer(s)),
        None => None,
    })
}

/// A scored dismax whose disjuncts are all terms, as [`DisMaxBulk`]; `None`
/// when it is not that shape (the scorer tree runs it).
fn bulk_dismax<'a>(
    ctx: &LeafContext<'a>,
    d: &crate::query::DisjunctionMaxQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<Option<Bulk<'a>>>> {
    if !mode.needs_scores() {
        return Ok(None);
    }
    let mut legs = Vec::with_capacity(d.disjuncts.len());
    for disjunct in &d.disjuncts {
        match term_leg(ctx, disjunct, boost, mode)? {
            TermForm::Leg(leg) => legs.push(*leg),
            TermForm::Absent => {}
            TermForm::Other => return Ok(None),
        }
    }
    Ok(Some(match legs.len() {
        0 => None,
        // `DisjunctionMaxQuery`'s one-scorer case: the scorer itself.
        1 => Some(Bulk::Term(
            Box::new(legs.pop().expect("one leg")),
            DocScores::default(),
        )),
        _ => Some(Bulk::DisMax(legs, Box::new(DisMaxBulk::new(d.tie_breaker)))),
    }))
}

/// `subs.get(..).iterator().next().bulkScorer()`: the bulk scorer of the one
/// positive clause a boolean has in this segment. A term leg or a plain
/// scorer is used as built; a nested boolean gets its own bulk scorer, which
/// means building it again from its clause.
fn lone<'a>(
    ctx: &LeafContext<'a>,
    built: Child<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
) -> Result<Option<Bulk<'a>>> {
    // A nested boolean or dismax has its own bulk scorer, built from the
    // clause again.
    let nested = match clause {
        Clause::Boolean(_) | Clause::DisjunctionMax(_) => true,
        Clause::Boost(b) => matches!(
            boost_chain(b).1,
            Clause::Boolean(_) | Clause::DisjunctionMax(_)
        ),
        _ => false,
    };
    Ok(match built {
        Child::Leg(leg) => Some(Bulk::Term(leg, DocScores::default())),
        Child::Scorer(_) if nested => return bulk_clause(ctx, clause, boost, mode),
        Child::Scorer(s) => Some(Bulk::scorer(s)),
    })
}

/// `BooleanWeight.bulkScorer`: `BooleanQuery.rewrite`'s single-clause
/// unwrapping, then `BooleanScorerSupplier.booleanScorer()`, then
/// `DefaultBulkScorer` over the scorer tree when that declines.
pub(crate) fn bulk_boolean<'a>(
    ctx: &LeafContext<'a>,
    q: &BooleanQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<Bulk<'a>>> {
    let clauses = q.must.len() + q.filter.len() + q.should.len() + q.must_not.len();
    if clauses == 1 {
        // `BooleanQuery.rewrite`: a lone `MUST` is its clause only when no
        // `SHOULD` is required -- with `minimum_should_match` 1 and no
        // `SHOULD` clause the query matches nothing.
        if let [only] = &q.must[..] {
            if q.minimum_should_match == 0 {
                return bulk_clause(ctx, only, boost, mode);
            }
        }
        if let [only] = &q.should[..] {
            if q.minimum_should_match <= 1 {
                return bulk_clause(ctx, only, boost, mode);
            }
        }
    }

    let child_top_level = q.must.len() + q.should.len() == 1;
    let mut must = Vec::with_capacity(q.must.len());
    for c in &q.must {
        match child(ctx, c, boost, mode, child_top_level)? {
            Some(s) => must.push((s, c)),
            None => return Ok(None),
        }
    }
    let mut filter = Vec::with_capacity(q.filter.len());
    for c in &q.filter {
        match child(ctx, c, boost, Mode::NoScores, false)? {
            Some(s) => filter.push((s, c)),
            None => return Ok(None),
        }
    }
    let mut should = Vec::with_capacity(q.should.len());
    for c in &q.should {
        if let Some(s) = child(ctx, c, boost, mode, child_top_level)? {
            should.push((s, c));
        }
    }
    let mut must_not = Vec::with_capacity(q.must_not.len());
    for c in &q.must_not {
        if let Some(s) = child(ctx, c, boost, Mode::NoScores, false)? {
            must_not.push(s);
        }
    }

    // `BooleanWeight.scorerSupplier`'s per-segment adjustments.
    let mut msm = q.minimum_should_match;
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

    // `booleanScorer()`: the positive clauses' bulk scorer, if there is one.
    let required = must.len() + filter.len();
    let all_legs = |v: &[(Child<'a>, &Clause)]| v.iter().all(|(c, _)| c.is_leg());
    let positive: Option<Bulk<'a>> = if required == 0 {
        if should.len() == 1 {
            let (c, clause) = should.pop().expect("one optional clause");
            lone(ctx, c, clause, boost, mode)?
        } else if msm <= 1 && all_legs(&should) {
            let mut legs: Vec<TermLeg<'a>> = std::mem::take(&mut should)
                .into_iter()
                .map(|(c, _)| c.into_leg())
                .collect();
            let state = MaxScore::new(&mut legs);
            Some(Bulk::Disjunction(legs, None, state))
        } else if msm <= 1 && mode == Mode::TopScores {
            // `optionalBulkScorer`: `MaxScoreBulkScorer` over whatever the
            // clauses are.
            let mut legs = scorer_legs(std::mem::take(&mut should), mode)?;
            let state = MaxScore::new(&mut legs);
            Some(Bulk::ScorerDisjunction(legs, None, state))
        } else {
            None
        }
    } else if must.is_empty() && should.len() > 1 && msm >= 1 {
        if mode == Mode::TopScores && msm == 1 && all_legs(&should) {
            let filter_scorer = filter_of(std::mem::take(&mut filter));
            let mut legs: Vec<TermLeg<'a>> = std::mem::take(&mut should)
                .into_iter()
                .map(|(c, _)| c.into_leg())
                .collect();
            let state = MaxScore::new(&mut legs);
            Some(Bulk::Disjunction(legs, Some(filter_scorer), state))
        } else if mode == Mode::TopScores && msm == 1 {
            let filter_scorer = filter_of(std::mem::take(&mut filter));
            let mut legs = scorer_legs(std::mem::take(&mut should), mode)?;
            let state = MaxScore::new(&mut legs);
            Some(Bulk::ScorerDisjunction(legs, Some(filter_scorer), state))
        } else {
            None
        }
    } else if required > 0 && should.is_empty() && msm == 0 {
        if required == 1 {
            if let Some((c, clause)) = must.pop() {
                lone(ctx, c, clause, boost, mode)?
            } else {
                let (c, _) = filter.pop().expect("one filter clause");
                // `disableScoring(filter.bulkScorer())`: its matches score 0.
                Some(match c {
                    Child::Leg(leg) => Bulk::Term(leg, DocScores::default()),
                    Child::Scorer(s) if mode.needs_scores() => {
                        Bulk::scorer(Box::new(ZeroScorer(s)))
                    }
                    Child::Scorer(s) => Bulk::scorer(s),
                })
            }
        } else if all_legs(&must) && all_legs(&filter) {
            let mut legs: Vec<TermLeg<'a>> = std::mem::take(&mut must)
                .into_iter()
                .chain(std::mem::take(&mut filter))
                .map(|(c, _)| c.into_leg())
                .collect();
            legs.sort_by_key(|l| l.cost);
            let state = ConjunctionBulk::new(legs.len());
            Some(Bulk::Conjunction(legs, state))
        } else if mode.needs_scores() && must.len() > 1 {
            // `BlockMaxConjunctionBulkScorer` over any clauses, when none is
            // two-phase: the filters join it as constant-0 scorers.
            let mut scorers: Vec<(i64, BoxScorer<'a>)> = std::mem::take(&mut must)
                .into_iter()
                .map(|(c, _)| (c.cost(), c.into_scorer(mode)))
                .chain(std::mem::take(&mut filter).into_iter().map(|(c, _)| {
                    let s = c.into_scorer(Mode::NoScores);
                    let zero: BoxScorer<'a> = Box::new(ZeroScorer(s));
                    (zero.cost(), zero)
                }))
                .collect();
            if scorers.iter().any(|(_, s)| s.two_phase()) {
                // Not this bulk scorer's shape; rebuild as the scorer tree.
                return bulk_boolean_tree(ctx, q, boost, mode);
            }
            scorers.sort_by_key(|(c, _)| *c);
            let scorers: Vec<BoxScorer<'a>> = scorers.into_iter().map(|(_, s)| s).collect();
            let state = ConjunctionBulk::new(scorers.len());
            Some(Bulk::ScorerConjunction(scorers, state))
        } else {
            None
        }
    } else if required > 0
        && msm == 0
        && mode.needs_scores()
        && all_legs(&must)
        && all_legs(&filter)
        && all_legs(&should)
    {
        let mut req: Vec<TermLeg<'a>> = std::mem::take(&mut must)
            .into_iter()
            .chain(std::mem::take(&mut filter))
            .map(|(c, _)| c.into_leg())
            .collect();
        req.sort_by_key(|l| l.cost);
        let opt: Vec<TermLeg<'a>> = std::mem::take(&mut should)
            .into_iter()
            .map(|(c, _)| c.into_leg())
            .collect();
        let state = Box::new(ReqOptBulk::new(req.len()));
        Some(Bulk::ReqOpt(req, opt, state))
    } else if required > 0 && msm >= 1 && !should.is_empty() && mode == Mode::TopScores {
        // Our own: Lucene declines a bulk scorer here and runs
        // `ConjunctionScorer(req, opt)` a document at a time, never skipping.
        // The same two scorers as a block-max conjunction score identically
        // (`ConjunctionScorer.score` is the same `f64` sum) and drop blocks
        // whose summed maxima cannot compete.
        let (req, opt) = super::build::req_and_opt(
            std::mem::take(&mut must)
                .into_iter()
                .map(|(c, _)| c)
                .collect(),
            std::mem::take(&mut filter)
                .into_iter()
                .map(|(c, _)| c)
                .collect(),
            std::mem::take(&mut should)
                .into_iter()
                .map(|(c, _)| c)
                .collect(),
            msm,
            mode,
        )?;
        if req.two_phase() || opt.two_phase() {
            return bulk_boolean_tree(ctx, q, boost, mode);
        }
        let mut scorers = vec![req, opt];
        scorers.sort_by_key(|s| s.cost());
        Some(Bulk::ScorerConjunction(scorers, ConjunctionBulk::new(2)))
    } else {
        None
    };

    if let Some(positive) = positive {
        if must_not.is_empty() {
            return Ok(Some(positive));
        }
        let mut excluded: Vec<BoxScorer<'a>> = must_not
            .into_iter()
            .map(|c| c.into_scorer(Mode::NoScores))
            .collect();
        let excl: BoxScorer<'a> = if excluded.len() == 1 {
            excluded.pop().expect("one prohibited clause")
        } else {
            Box::new(DisjunctionScorer::new(excluded, Combine::Sum, false))
        };
        return Ok(Some(Bulk::ReqExcl(Box::new(positive), excl)));
    }

    // No bulk scorer for this shape: the scorer tree, `getInternal`'s.
    let strip = |v: Vec<(Child<'a>, &Clause)>| -> Vec<Child<'a>> {
        v.into_iter().map(|(c, _)| c).collect()
    };
    Ok(compose(
        strip(must),
        strip(filter),
        strip(should),
        must_not,
        msm,
        mode,
        true,
    )?
    .map(Bulk::scorer))
}

/// The scorer tree for `q`, driven by `DefaultBulkScorer`: what
/// `booleanScorer()` returning `null` means.
fn bulk_boolean_tree<'a>(
    ctx: &LeafContext<'a>,
    q: &BooleanQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<Bulk<'a>>> {
    Ok(super::build::build_boolean(ctx, q, boost, mode, true)?.map(Bulk::scorer))
}

/// `filteredOptionalBulkScorer`'s filter: the `FILTER` clauses as one
/// iterator.
fn filter_of<'a>(filter: Vec<(Child<'a>, &Clause)>) -> BoxScorer<'a> {
    let mut filters: Vec<BoxScorer<'a>> = filter
        .into_iter()
        .map(|(c, _)| c.into_scorer(Mode::NoScores))
        .collect();
    if filters.len() == 1 {
        filters.pop().expect("one filter")
    } else {
        Box::new(ConjunctionScorer::new(filters, Vec::new()))
    }
}

/// The `SHOULD` clauses as [`MaxScore`] clauses, positioned on their first
/// match.
fn scorer_legs<'a>(should: Vec<(Child<'a>, &Clause)>, mode: Mode) -> Result<Vec<ScorerLeg<'a>>> {
    should
        .into_iter()
        .map(|(c, _)| Ok(ScorerLeg::new(c.into_scorer(mode))))
        .collect()
}

impl<'a> ScorerLeg<'a> {
    pub(crate) fn new(s: BoxScorer<'a>) -> Self {
        let cost = s.cost();
        let doc = s.doc_id();
        ScorerLeg {
            s,
            doc,
            cost,
            max_window_score: 0.0,
        }
    }
}

/// A [`MaxScore`] clause that is any scorer: `DisiWrapper` around it,
/// iterating `scorer.iterator()` -- a two-phase scorer's *exact* view, as
/// `TwoPhaseIterator.asDocIdSetIterator` gives `MaxScoreBulkScorer`.
pub(crate) struct ScorerLeg<'a> {
    s: BoxScorer<'a>,
    doc: i32,
    cost: i64,
    max_window_score: f32,
}

impl BulkLeg for ScorerLeg<'_> {
    fn doc_id(&self) -> i32 {
        self.s.doc_id()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        super::exact_advance(&mut *self.s, target)
    }
    fn next_doc(&mut self) -> Result<i32> {
        super::exact_next(&mut *self.s)
    }
    fn score(&mut self) -> Result<f32> {
        super::Scorer::score(&mut *self.s)
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        super::Scorer::advance_shallow(&mut *self.s, target)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        super::Scorer::max_score(&mut *self.s, up_to)
    }
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()> {
        // `Scorer`'s, over the exact iterator -- not `BulkLeg for BoxScorer`,
        // which method resolution would pick and which walks the
        // approximation.
        super::Scorer::next_docs_and_scores(&mut *self.s, up_to, live_docs, out)
    }
}

impl MaxScoreLeg for ScorerLeg<'_> {
    fn cur(&self) -> i32 {
        self.doc
    }
    fn set_cur(&mut self, doc: i32) {
        self.doc = doc;
    }
    fn leg_cost(&self) -> i64 {
        self.cost
    }
    fn window_max(&self) -> f32 {
        self.max_window_score
    }
    fn set_window_max(&mut self, score: f32) {
        self.max_window_score = score;
    }
}

impl crate::bulk_scorer::BulkLeg for BoxScorer<'_> {
    fn doc_id(&self) -> i32 {
        super::Scorer::doc_id(&**self)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        super::Scorer::advance(&mut **self, target)
    }
    fn next_doc(&mut self) -> Result<i32> {
        super::Scorer::next_doc(&mut **self)
    }
    fn score(&mut self) -> Result<f32> {
        super::Scorer::score(&mut **self)
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        super::Scorer::advance_shallow(&mut **self, target)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        super::Scorer::max_score(&mut **self, up_to)
    }
    /// `Scorer.nextDocsAndScores`. Only non-two-phase scorers reach a bulk
    /// scorer as a `BulkLeg`, so the exact iterator is the approximation.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()> {
        debug_assert!(!self.two_phase());
        super::Scorer::next_docs_and_scores(&mut **self, up_to, live_docs, out)
    }
}

/// Scores `bulk` over the whole segment.
pub(crate) fn score_segment<C: ScoringCollector + ?Sized>(
    bulk: &mut Bulk<'_>,
    mode: Mode,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
) -> Result<()> {
    bulk.score(mode, live_docs, collector, 0, NO_MORE_DOCS)?;
    Ok(())
}
