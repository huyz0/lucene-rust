//! `WANDScorer`: a disjunction that, in `TOP_SCORES`, keeps the clauses whose
//! summed block maxima cannot reach the minimum competitive score in a `tail`
//! and never iterates them, and that also implements `minimum_should_match`.

use super::disjunction::{Disi, DisiQueue};
use super::{exact_advance, BoxScorer, Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::sum_upper_bound;
use crate::Result;

/// `WANDScorer.FLOAT_MANTISSA_BITS`.
const FLOAT_MANTISSA_BITS: usize = 24;
/// `WANDScorer.MAX_SCALED_SCORE`.
const MAX_SCALED_SCORE: i64 = (1 << 24) - 1;

/// `Math.getExponent(double)` for a finite, non-zero value.
fn exponent(d: f64) -> i32 {
    // The biased exponent is 11 bits, so the cast is lossless.
    ((d.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// `WANDScorer.scalingFactor`.
pub(crate) fn scaling_factor(f: f32) -> i32 {
    if f == 0.0 {
        scaling_factor(f32::from_bits(1)) + 1
    } else if f.is_infinite() {
        scaling_factor(f32::MAX) - 1
    } else {
        FLOAT_MANTISSA_BITS as i32 - 1 - exponent(f64::from(f))
    }
}

/// `Math.scalb(d, n)` for the range a float's scaling factor spans.
///
/// `2^n` is built from its exponent bits while it is a normal double -- the
/// same exact power `powi` returns, without the libm call on every document
/// WAND scales a score for.
fn scalb(d: f64, n: i32) -> f64 {
    let pow = if (-1022..=1023).contains(&n) {
        // ARITH: -1022 <= n <= 1023, so the biased exponent is 1..=2046.
        f64::from_bits(((n + 1023) as u64) << 52)
    } else {
        2f64.powi(n)
    };
    d * pow
}

/// `WANDScorer.scaleMaxScore`: rounds up.
pub(crate) fn scale_max_score(max_score: f32, scaling_factor: i32) -> i64 {
    let scaled = scalb(f64::from(max_score), scaling_factor);
    if scaled > MAX_SCALED_SCORE as f64 {
        return MAX_SCALED_SCORE;
    }
    scaled.ceil() as i64
}

/// `WANDScorer.scaleMinScore`: rounds down.
fn scale_min_score(min_score: f32, scaling_factor: i32) -> i64 {
    scalb(f64::from(min_score), scaling_factor).floor() as i64
}

/// `ScorerUtil.costWithMinShouldMatch`: the sum of the `n - msm + 1` smallest
/// costs.
pub(crate) fn cost_with_min_should_match(costs: &[i64], min_should_match: usize) -> i64 {
    let mut sorted = costs.to_vec();
    sorted.sort_unstable();
    let keep = (costs.len() + 1).saturating_sub(min_should_match);
    sorted
        .iter()
        .take(keep)
        .fold(0i64, |a, &c| a.saturating_add(c))
}

pub(crate) struct WandScorer<'a> {
    subs: Vec<Disi<'a>>,
    scaling_factor: i32,
    min_competitive_score: i64,
    /// Sub-scorers on `doc`.
    lead: Vec<usize>,
    doc: i32,
    lead_score: f64,
    /// Sub-scorers ahead of `doc`, by document.
    head: DisiQueue,
    /// Sub-scorers behind `doc`, a max-heap by scaled maximum score.
    tail: Vec<usize>,
    tail_max_score: i64,
    cost: i64,
    /// The last document the maxima in `subs` are valid for.
    up_to: i32,
    min_should_match: usize,
    freq: usize,
    top_scores: bool,
    lead_cost: i64,
}

impl<'a> WandScorer<'a> {
    pub(crate) fn new(
        scorers: Vec<BoxScorer<'a>>,
        min_should_match: usize,
        top_scores: bool,
        lead_cost: i64,
    ) -> Result<Self> {
        debug_assert!(min_should_match < scorers.len());
        let mut subs: Vec<Disi<'a>> = scorers.into_iter().map(Disi::new).collect();
        let scaling_factor = if top_scores {
            let mut sum = 0.0f64;
            for w in &mut subs {
                w.scorer.advance_shallow(0)?;
                sum += f64::from(w.scorer.max_score(NO_MORE_DOCS)?);
            }
            scaling_factor(sum_upper_bound(sum, subs.len()) as f32)
        } else {
            0
        };
        let costs: Vec<i64> = subs.iter().map(|w| w.cost).collect();
        let n = subs.len();
        Ok(Self {
            scaling_factor,
            min_competitive_score: 0,
            // `addUnpositionedLead` for every scorer.
            lead: (0..n).collect(),
            freq: n,
            doc: -1,
            lead_score: 0.0,
            head: DisiQueue::with_capacity(n),
            tail: Vec::with_capacity(n),
            tail_max_score: 0,
            cost: cost_with_min_should_match(&costs, min_should_match),
            up_to: -1,
            min_should_match,
            top_scores,
            lead_cost,
            subs,
        })
    }

    fn add_lead(&mut self, i: usize) -> Result<()> {
        self.lead.push(i);
        self.freq += 1;
        if self.top_scores {
            self.lead_score += f64::from(self.subs[i].scorer.score()?);
        }
        Ok(())
    }

    fn push_back_leads(&mut self, target: i32) -> Result<()> {
        for k in 0..self.lead.len() {
            let s = self.lead[k];
            if let Some(evicted) = self.insert_tail_with_overflow(s) {
                let w = &mut self.subs[evicted];
                w.doc = exact_advance(&mut *w.scorer, target)?;
                self.head.push(&self.subs, evicted);
            }
        }
        self.lead.clear();
        Ok(())
    }

    fn advance_head(&mut self, target: i32) -> Result<Option<usize>> {
        let mut head_top = self.head.top();
        while let Some(top) = head_top {
            if self.subs[top].doc >= target {
                break;
            }
            match self.insert_tail_with_overflow(top) {
                Some(evicted) => {
                    let w = &mut self.subs[evicted];
                    w.doc = exact_advance(&mut *w.scorer, target)?;
                    head_top = Some(self.head.replace_top(&self.subs, evicted));
                }
                None => {
                    self.head.pop(&self.subs);
                    head_top = self.head.top();
                }
            }
        }
        Ok(head_top)
    }

    fn advance_tail_one(&mut self, i: usize) -> Result<()> {
        let doc = self.doc;
        let w = &mut self.subs[i];
        w.doc = exact_advance(&mut *w.scorer, doc)?;
        if w.doc == doc {
            self.add_lead(i)
        } else {
            self.head.push(&self.subs, i);
            Ok(())
        }
    }

    fn advance_tail(&mut self) -> Result<()> {
        let top = self.pop_tail();
        self.advance_tail_one(top)
    }

    fn update_max_scores(&mut self, target: i32) -> Result<()> {
        let mut new_up_to = NO_MORE_DOCS;
        for k in 0..self.head.len() {
            let i = self.head.members()[k];
            let w = &mut self.subs[i];
            if w.doc <= new_up_to && w.cost <= self.lead_cost {
                new_up_to = new_up_to.min(w.scorer.advance_shallow(w.doc)?);
            }
        }
        if new_up_to == NO_MORE_DOCS
            && !self.tail.is_empty()
            && self.subs[self.tail[0]].cost <= self.lead_cost
        {
            new_up_to = self.subs[self.tail[0]].scorer.advance_shallow(target)?;
            if let Some(top) = self.head.top() {
                new_up_to = new_up_to.max(self.subs[top].doc);
            }
        }
        self.up_to = new_up_to;

        for k in 0..self.head.len() {
            let i = self.head.members()[k];
            let w = &mut self.subs[i];
            if w.doc <= self.up_to {
                w.scaled_max_score =
                    scale_max_score(w.scorer.max_score(new_up_to)?, self.scaling_factor);
            }
        }

        self.tail_max_score = 0;
        for k in 0..self.tail.len() {
            let i = self.tail[k];
            let w = &mut self.subs[i];
            w.scorer.advance_shallow(target)?;
            w.scaled_max_score =
                scale_max_score(w.scorer.max_score(self.up_to)?, self.scaling_factor);
            let scaled = w.scaled_max_score;
            self.up_heap_max_score(k);
            self.tail_max_score += scaled;
        }

        while !self.tail.is_empty() && self.tail_max_score >= self.min_competitive_score {
            let w = self.pop_tail();
            let sub = &mut self.subs[w];
            sub.doc = exact_advance(&mut *sub.scorer, target)?;
            self.head.push(&self.subs, w);
        }
        Ok(())
    }

    fn move_to_next_block(&mut self, mut target: i32) -> Result<()> {
        while self.up_to < NO_MORE_DOCS {
            match self.head.top() {
                None => {
                    target = target.max(self.up_to.saturating_add(1));
                    self.update_max_scores(target)?;
                }
                Some(top) if self.subs[top].doc > self.up_to => {
                    self.update_max_scores(target)?;
                    break;
                }
                Some(_) => break,
            }
        }
        Ok(())
    }

    fn move_to_next_candidate(&mut self) -> Result<()> {
        let first = self
            .head
            .pop(&self.subs)
            .expect("a candidate is on the head");
        self.lead.clear();
        self.lead.push(first);
        self.freq = 1;
        if self.top_scores {
            self.lead_score = f64::from(self.subs[first].scorer.score()?);
        }
        while let Some(top) = self.head.top() {
            if self.subs[top].doc != self.doc {
                break;
            }
            let i = self.head.pop(&self.subs).expect("non-empty");
            self.add_lead(i)?;
        }
        Ok(())
    }

    fn advance_all_tail(&mut self) -> Result<()> {
        for k in (0..self.tail.len()).rev() {
            let i = self.tail[k];
            self.advance_tail_one(i)?;
        }
        self.tail.clear();
        self.tail_max_score = 0;
        Ok(())
    }

    fn scaled_lead_score(&self) -> i64 {
        scale_max_score(
            sum_upper_bound(self.lead_score, FLOAT_MANTISSA_BITS) as f32,
            self.scaling_factor,
        )
    }

    fn insert_tail_with_overflow(&mut self, s: usize) -> Option<usize> {
        let scaled = self.subs[s].scaled_max_score;
        if self.tail_max_score + scaled < self.min_competitive_score
            || self.tail.len() + 1 < self.min_should_match
        {
            self.add_tail(s);
            self.tail_max_score += scaled;
            None
        } else if self.tail.is_empty() {
            Some(s)
        } else {
            let top = self.tail[0];
            if !self.greater_max_score(top, s) {
                return Some(s);
            }
            self.tail[0] = s;
            self.down_heap_max_score();
            self.tail_max_score = self.tail_max_score - self.subs[top].scaled_max_score + scaled;
            Some(top)
        }
    }

    fn add_tail(&mut self, s: usize) {
        self.tail.push(s);
        self.up_heap_max_score(self.tail.len() - 1);
    }

    fn pop_tail(&mut self) -> usize {
        let result = self.tail[0];
        let last = self.tail.pop().expect("non-empty tail");
        if !self.tail.is_empty() {
            self.tail[0] = last;
            self.down_heap_max_score();
        }
        self.tail_max_score -= self.subs[result].scaled_max_score;
        result
    }

    fn greater_max_score(&self, a: usize, b: usize) -> bool {
        let (w1, w2) = (&self.subs[a], &self.subs[b]);
        w1.scaled_max_score > w2.scaled_max_score
            || (w1.scaled_max_score == w2.scaled_max_score && w1.cost < w2.cost)
    }

    fn up_heap_max_score(&mut self, mut i: usize) {
        let node = self.tail[i];
        while i > 0 {
            let j = (i - 1) / 2;
            if !self.greater_max_score(node, self.tail[j]) {
                break;
            }
            self.tail[i] = self.tail[j];
            i = j;
        }
        self.tail[i] = node;
    }

    fn down_heap_max_score(&mut self) {
        let size = self.tail.len();
        let mut i = 0;
        let node = self.tail[0];
        loop {
            let left = 2 * i + 1;
            if left >= size {
                break;
            }
            let right = left + 1;
            let j = if right < size && self.greater_max_score(self.tail[right], self.tail[left]) {
                right
            } else {
                left
            };
            if !self.greater_max_score(self.tail[j], node) {
                break;
            }
            self.tail[i] = self.tail[j];
            i = j;
        }
        self.tail[i] = node;
    }
}

impl Scorer for WandScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        self.push_back_leads(target)?;
        let mut head_top = self.advance_head(target)?;
        if self.top_scores && head_top.is_none_or(|t| self.subs[t].doc > self.up_to) {
            self.move_to_next_block(target)?;
            head_top = self.head.top();
        }
        self.doc = head_top.map_or(NO_MORE_DOCS, |t| self.subs[t].doc);
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.cost
    }

    fn two_phase(&self) -> bool {
        true
    }

    fn matches(&mut self) -> Result<bool> {
        self.move_to_next_candidate()?;
        let mut scaled_lead = if self.top_scores {
            self.scaled_lead_score()
        } else {
            0
        };
        while scaled_lead < self.min_competitive_score || self.freq < self.min_should_match {
            if scaled_lead + self.tail_max_score < self.min_competitive_score
                || self.freq + self.tail.len() < self.min_should_match
            {
                return Ok(false);
            }
            let before = self.lead.len();
            self.advance_tail()?;
            if self.top_scores && self.lead.len() != before {
                scaled_lead = self.scaled_lead_score();
            }
        }
        Ok(true)
    }

    fn match_cost(&self) -> f32 {
        self.subs.len() as f32
    }

    fn score(&mut self) -> Result<f32> {
        self.advance_all_tail()?;
        let mut score = self.lead_score;
        if !self.top_scores {
            for k in 0..self.lead.len() {
                let i = self.lead[k];
                score += f64::from(self.subs[i].scorer.score()?);
            }
        }
        Ok(score as f32)
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        for w in &mut self.subs {
            if w.scorer.doc_id() < target {
                w.scorer.advance_shallow(target)?;
            }
        }
        Ok(if target <= self.up_to {
            self.up_to
        } else {
            NO_MORE_DOCS
        })
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut sum = 0.0f64;
        for w in &mut self.subs {
            if w.scorer.doc_id() <= up_to {
                sum += f64::from(w.scorer.max_score(up_to)?);
            }
        }
        Ok(sum_upper_bound(sum, self.subs.len()) as f32)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.min_competitive_score = scale_min_score(min, self.scaling_factor);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalb_builds_the_power_powi_does() {
        for n in -1100..=1100 {
            for d in [1.0f64, 0.3, 17.25, f64::from(f32::MAX), 1e-30] {
                assert_eq!(
                    scalb(d, n).to_bits(),
                    (d * 2f64.powi(n)).to_bits(),
                    "{d} {n}"
                );
            }
        }
    }
}
