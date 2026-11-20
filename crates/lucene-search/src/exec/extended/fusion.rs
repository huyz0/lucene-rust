//! The scorers that fuse or fold other queries' scores: `IndriAndQuery`,
//! `LogOddsFusionQuery` and the Bayesian score query.

use super::*;

// ---------------------------------------------------------------------------
// IndriAndQuery
// ---------------------------------------------------------------------------

/// `IndriAndScorer`: the disjunction of its clauses. Its score folds only
/// sub-scorers that are themselves `IndriScorer`s (a nested composite
/// `IndriAndScorer`) -- a term's `TermScorer` is not one, so it contributes
/// neither a score nor a boost, exactly as `scoreDoc`'s `instanceof` check
/// has it.
struct IndriAndScorer<'a> {
    approx: DisiApprox<BoxScorer<'a>>,
    boost: f32,
    /// The sub-scorers that are `IndriScorer`s, with their boosts: fixed
    /// when the scorer is built, so the per-document `instanceof` walk visits
    /// only these (none, for an Indri query of plain terms, whose every
    /// document then scores `0`).
    indri: Vec<(usize, f32)>,
}

impl<'a> IndriAndScorer<'a> {
    fn new(subs: Vec<BoxScorer<'a>>, boost: f32) -> Self {
        let indri = subs
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.indri_boost().map(|b| (i, b)))
            .collect();
        IndriAndScorer {
            approx: DisiApprox::new(subs, i64::MAX),
            boost,
            indri,
        }
    }

    /// `scoreDoc(subScorers, docId)`.
    fn score_doc(&mut self, doc: i32) -> Result<f32> {
        let mut score = 0.0f64;
        let mut boost_sum = 0.0f64;
        for &(i, b) in &self.indri {
            let sub = &mut self.approx.subs[i];
            // `score()` on the document, `smoothingScore(docId)` otherwise;
            // both are `scoreDoc` for a composite.
            let s = sub.smoothing_score(doc)?;
            score += f64::from(s) * f64::from(b);
            boost_sum += f64::from(b);
        }
        Ok(if boost_sum == 0.0 {
            0.0
        } else {
            (score / boost_sum) as f32
        })
    }
}

impl Scorer for IndriAndScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.approx.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.approx.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.approx.advance(target)
    }
    fn cost(&self) -> i64 {
        self.approx.cost()
    }
    fn score(&mut self) -> Result<f32> {
        let doc = self.doc_id();
        self.score_doc(doc)
    }
    /// `IndriDisjunctionScorer.getMaxScore`: `0`.
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
    fn indri_boost(&self) -> Option<f32> {
        Some(self.boost)
    }
    fn smoothing_score(&mut self, doc: i32) -> Result<f32> {
        self.score_doc(doc)
    }
}

/// `IndriAndWeight`: every clause's weight is created with boost `1`; no
/// scorer is `null`, one is itself, more are an [`IndriAndScorer`] carrying
/// the query's boost.
pub(super) fn indri_and<'a>(
    ctx: &LeafContext<'a>,
    q: &IndriAndQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let mut subs = Vec::new();
    for c in &q.clauses {
        if let Some(s) = build::build(ctx, c, 1.0, mode, false)? {
            subs.push(s);
        }
    }
    Ok(match subs.len() {
        0 => None,
        1 => subs.pop(),
        _ => Some(Box::new(IndriAndScorer::new(subs, boost))),
    })
}

// ---------------------------------------------------------------------------
// LogOddsFusionQuery
// ---------------------------------------------------------------------------

const CLAMP_MIN: f32 = 1e-7;
const CLAMP_MAX: f32 = 1.0 - 1e-7;

/// `LogOddsFusionScorer.logit`.
pub(crate) fn logit(p: f32) -> f32 {
    let clamped = p.clamp(CLAMP_MIN, CLAMP_MAX);
    f64::from(clamped / (1.0 - clamped)).ln() as f32
}

/// `LogOddsFusionScorer.sigmoid` / `BayesianScoreQuery.sigmoid`.
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        (1.0 / (1.0 + (-f64::from(x)).exp())) as f32
    } else {
        let e = f64::from(x).exp();
        (e / (1.0 + e)) as f32
    }
}

/// `LogOddsFusionScorer.softplus`.
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        return x;
    }
    f64::from(x).exp().ln_1p() as f32
}

/// The fusion's parameters, shared by the scorer.
struct Fusion {
    total_clauses: usize,
    scaling: f32,
    weights: Option<Vec<f32>>,
    bounds: Option<(Vec<f32>, Vec<f32>)>,
}

impl Fusion {
    /// `gateLogit(rawLogit, signalIndex)`.
    fn gate(&self, raw: f32, i: usize) -> f32 {
        if let Some((min, max)) = &self.bounds {
            let range = max[i] - min[i];
            if range > 0.0 {
                return ((raw - min[i]) / range).clamp(0.0, 1.0);
            }
            return 0.5;
        }
        softplus(raw)
    }

    fn finish(&self, sum: f64) -> f32 {
        let scaled = if self.weights.is_some() {
            sum as f32 * self.scaling
        } else {
            (sum / self.total_clauses as f64) as f32 * self.scaling
        };
        sigmoid(scaled)
    }
}

struct LogOddsScorer<'a> {
    approx: DisiApprox<BoxScorer<'a>>,
    fusion: Fusion,
    two_phase: bool,
    list: Vec<usize>,
    verified: Vec<usize>,
}

impl Scorer for LogOddsScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.approx.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.approx.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.approx.advance(target)
    }
    fn cost(&self) -> i64 {
        self.approx.cost()
    }
    fn two_phase(&self) -> bool {
        self.two_phase
    }
    /// `DisjunctionScorer.TwoPhase.matches`: the members on the document
    /// that match, the ones without a two-phase view first.
    fn matches(&mut self) -> Result<bool> {
        self.approx.top_list(&mut self.list);
        self.verified.clear();
        for &w in self.list.iter().rev() {
            if !self.approx.subs[w].two_phase() {
                self.verified.push(w);
            }
        }
        for &w in &self.list {
            if self.approx.subs[w].two_phase() && self.approx.subs[w].matches()? {
                self.verified.push(w);
            }
        }
        Ok(!self.verified.is_empty())
    }
    fn score(&mut self) -> Result<f32> {
        if !self.two_phase {
            self.approx.top_list(&mut self.list);
            self.verified.clone_from(&self.list);
        }
        let mut sum = 0.0f64;
        for k in 0..self.verified.len() {
            let w = self.verified[k];
            let sub = self.approx.subs[w].score()?;
            // `scorerIndexMap` only exists with weights; without, every
            // clause gates with index 0's bounds.
            let idx = if self.fusion.weights.is_some() { w } else { 0 };
            let gated = self.fusion.gate(logit(sub), idx);
            match &self.fusion.weights {
                Some(ws) => sum += f64::from(ws[w] * gated),
                None => sum += f64::from(gated),
            }
        }
        Ok(self.fusion.finish(sum))
    }
    /// `LogOddsFusionScorer.getMaxScore`.
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut sum = 0.0f64;
        for i in 0..self.approx.subs.len() {
            if self.approx.subs[i].doc_id() <= up_to {
                let m = self.approx.subs[i].max_score(up_to)?;
                let gated = self.fusion.gate(logit(m), i);
                match &self.fusion.weights {
                    Some(ws) => sum += f64::from(ws[i] * gated),
                    None => sum += f64::from(gated),
                }
            }
        }
        Ok(self.fusion.finish(sum))
    }
}

/// `LogOddsFusionWeight.scorerSupplier`: the clauses present in the segment;
/// one is itself, more are a [`LogOddsScorer`] over the active clauses'
/// weights and bounds, with `n^alpha` over **every** clause.
pub(super) fn log_odds<'a>(
    ctx: &LeafContext<'a>,
    q: &LogOddsFusionQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let q = rewrite_log_odds(q);
    let q = match q {
        Rewritten::None => return Ok(None),
        Rewritten::One(c) => return build::build(ctx, &c, boost, mode, false),
        Rewritten::Fusion(q) => q,
    };
    let mut subs = Vec::new();
    let mut weights = q.weights.as_ref().map(|_| Vec::new());
    let mut bounds = q.logit_bounds.as_ref().map(|_| (Vec::new(), Vec::new()));
    for (i, c) in q.clauses.iter().enumerate() {
        if let Some(s) = build::build(ctx, c, boost, mode, false)? {
            subs.push(s);
            if let (Some(w), Some(all)) = (weights.as_mut(), q.weights.as_ref()) {
                w.push(all[i]);
            }
            if let (Some((lo, hi)), Some((all_lo, all_hi))) =
                (bounds.as_mut(), q.logit_bounds.as_ref())
            {
                lo.push(all_lo[i]);
                hi.push(all_hi[i]);
            }
        }
    }
    if subs.len() <= 1 {
        return Ok(subs.pop());
    }
    let total = q.clauses.len();
    let fusion = Fusion {
        total_clauses: total,
        scaling: (total as f64).powf(f64::from(q.alpha)) as f32,
        weights,
        bounds,
    };
    let two_phase = subs.iter().any(|s| s.two_phase());
    Ok(Some(Box::new(LogOddsScorer {
        approx: DisiApprox::new(subs, i64::MAX),
        fusion,
        two_phase,
        list: Vec::new(),
        verified: Vec::new(),
    })))
}

pub(super) enum Rewritten {
    None,
    One(Clause),
    Fusion(LogOddsFusionQuery),
}

/// `LogOddsFusionQuery.rewrite`: no clause matches nothing, one is itself,
/// and `MatchNoDocsQuery` clauses drop out with their weights renormalised.
pub(super) fn rewrite_log_odds(q: &LogOddsFusionQuery) -> Rewritten {
    if q.clauses.is_empty() {
        return Rewritten::None;
    }
    if q.clauses.len() == 1 {
        return Rewritten::One(q.clauses[0].clone());
    }
    let keep: Vec<usize> = (0..q.clauses.len())
        .filter(|&i| !matches!(q.clauses[i], Clause::MatchNoDocs(_)))
        .collect();
    if keep.len() == q.clauses.len() {
        return Rewritten::Fusion(q.clone());
    }
    match keep.len() {
        0 => Rewritten::None,
        1 => Rewritten::One(q.clauses[keep[0]].clone()),
        _ => {
            let weights = q.weights.as_ref().map(|w| {
                let mut kept: Vec<f32> = keep.iter().map(|&i| w[i]).collect();
                let sum: f32 = kept.iter().sum();
                if sum > 0.0 {
                    for x in kept.iter_mut() {
                        *x /= sum;
                    }
                }
                kept
            });
            let bounds = q.logit_bounds.as_ref().map(|(lo, hi)| {
                (
                    keep.iter().map(|&i| lo[i]).collect(),
                    keep.iter().map(|&i| hi[i]).collect(),
                )
            });
            Rewritten::Fusion(LogOddsFusionQuery {
                clauses: keep.iter().map(|&i| q.clauses[i].clone()).collect(),
                alpha: q.alpha,
                weights,
                logit_bounds: bounds,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// BayesianScoreQuery
// ---------------------------------------------------------------------------

struct BayesianScorer<'a> {
    inner: BoxScorer<'a>,
    alpha: f32,
    beta: f32,
    logit_base_rate: f32,
}

impl BayesianScorer<'_> {
    fn transform(&self, s: f32) -> f32 {
        sigmoid(self.alpha * (s - self.beta) + self.logit_base_rate)
    }
}

impl Scorer for BayesianScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.inner.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.inner.advance(target)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    fn two_phase(&self) -> bool {
        self.inner.two_phase()
    }
    fn matches(&mut self) -> Result<bool> {
        self.inner.matches()
    }
    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
    fn score(&mut self) -> Result<f32> {
        let s = self.inner.score()?;
        Ok(self.transform(s))
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.inner.advance_shallow(target)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let m = self.inner.max_score(up_to)?;
        Ok(self.transform(m))
    }
    /// `BayesianScoreScorer.setMinCompetitiveScore`: the threshold mapped
    /// back through the inverse sigmoid.
    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if min > 0.0 && min < 1.0 {
            let clamped = min.clamp(1e-7, 1.0 - 1e-7);
            let logit_min = f64::from(clamped / (1.0 - clamped)).ln() as f32;
            let inner_min = (logit_min - self.logit_base_rate) / self.alpha + self.beta;
            self.inner.set_min_competitive_score(inner_min.max(0.0))?;
        }
        Ok(())
    }
    fn doc_id_run_end(&self) -> i32 {
        self.inner.doc_id_run_end()
    }
}

pub(super) fn bayesian<'a>(
    ctx: &LeafContext<'a>,
    q: &BayesianScoreQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(inner) = build::build(ctx, &q.query, boost, mode, top_level)? else {
        return Ok(None);
    };
    if !mode.needs_scores() {
        return Ok(Some(inner));
    }
    Ok(Some(Box::new(BayesianScorer {
        inner,
        alpha: q.alpha,
        beta: q.beta,
        logit_base_rate: q.logit_base_rate(),
    })))
}
