//! `ReqExclScorer` (`MUST_NOT`) and `ReqOptSumScorer` (`MUST` + optional
//! `SHOULD`).

use super::{BoxScorer, Scorer, NO_MORE_DOCS};
use crate::Result;

/// `ReqExclScorer.ADVANCE_COST`.
const ADVANCE_COST: f32 = 10.0;

/// `ReqExclScorer`: the required scorer's matches that the excluded one does
/// not match. Always two-phase: the exclusion is checked in `matches`.
pub(crate) struct ReqExclScorer<'a> {
    req: BoxScorer<'a>,
    excl: BoxScorer<'a>,
    match_cost: f32,
    /// Whether the required scorer's own match check runs first.
    req_first: bool,
}

impl<'a> ReqExclScorer<'a> {
    pub(crate) fn new(req: BoxScorer<'a>, excl: BoxScorer<'a>) -> Self {
        let mut match_cost = 2.0f32;
        if req.two_phase() {
            match_cost += req.match_cost();
        }
        let excl_match_cost = ADVANCE_COST
            + if excl.two_phase() {
                excl.match_cost()
            } else {
                0.0
            };
        let ratio = if req.cost() <= 0 {
            1.0
        } else if excl.cost() <= 0 {
            0.0
        } else {
            req.cost().min(excl.cost()) as f32 / req.cost() as f32
        };
        match_cost += ratio * excl_match_cost;
        let req_first =
            !req.two_phase() || (excl.two_phase() && req.match_cost() <= excl.match_cost());
        Self {
            req,
            excl,
            match_cost,
            req_first,
        }
    }
}

impl Scorer for ReqExclScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.req.doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        self.req.next_doc()
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        self.req.advance(target)
    }

    fn cost(&self) -> i64 {
        self.req.cost()
    }

    fn two_phase(&self) -> bool {
        true
    }

    fn matches(&mut self) -> Result<bool> {
        let doc = self.req.doc_id();
        let mut excl_doc = self.excl.doc_id();
        if excl_doc < doc {
            excl_doc = self.excl.advance(doc)?;
        }
        if excl_doc != doc {
            return self.req.matches();
        }
        // The cheaper check first: `matches` advances two-phase state, so
        // the order is part of the behaviour, not only of the cost.
        if self.req_first {
            if !self.req.matches()? {
                return Ok(false);
            }
            Ok(!self.excl.matches()?)
        } else {
            if self.excl.matches()? {
                return Ok(false);
            }
            self.req.matches()
        }
    }

    fn match_cost(&self) -> f32 {
        self.match_cost
    }

    fn score(&mut self) -> Result<f32> {
        self.req.score()
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.req.advance_shallow(target)
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        self.req.max_score(up_to)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.req.set_min_competitive_score(min)
    }
}

/// `ReqOptSumScorer`: the required scorer's matches, plus the optional
/// scorer's score where it matches too. In `TOP_SCORES` its approximation
/// skips blocks whose combined maxima cannot compete, and once the required
/// clause alone cannot reach the threshold the optional one becomes required.
pub(crate) struct ReqOptSumScorer<'a> {
    req: BoxScorer<'a>,
    opt: BoxScorer<'a>,
    top_scores: bool,
    min_score: f32,
    req_max_score: f32,
    opt_is_required: bool,
    up_to: i32,
    block_max: f32,
}

impl<'a> ReqOptSumScorer<'a> {
    pub(crate) fn new(
        mut req: BoxScorer<'a>,
        mut opt: BoxScorer<'a>,
        top_scores: bool,
    ) -> Result<Self> {
        let req_max_score = if top_scores {
            req.advance_shallow(0)?;
            opt.advance_shallow(0)?;
            req.max_score(NO_MORE_DOCS)?
        } else {
            f32::INFINITY
        };
        Ok(Self {
            req,
            opt,
            top_scores,
            min_score: 0.0,
            req_max_score,
            opt_is_required: false,
            up_to: -1,
            block_max: 0.0,
        })
    }

    fn move_to_next_block(&mut self, target: i32) -> Result<()> {
        self.up_to = self.advance_shallow(target)?;
        let req_block_max = self.req.max_score(self.up_to)?;
        self.block_max = self.max_score(self.up_to)?;
        self.opt_is_required = req_block_max < self.min_score;
        Ok(())
    }

    fn advance_impacts(&mut self, mut target: i32) -> Result<i32> {
        if target > self.up_to {
            self.move_to_next_block(target)?;
        }
        loop {
            if self.block_max >= self.min_score {
                return Ok(target);
            }
            if self.up_to == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            }
            target = self.up_to.saturating_add(1);
            self.move_to_next_block(target)?;
        }
    }

    fn advance_internal(&mut self, target: i32) -> Result<i32> {
        if target == NO_MORE_DOCS {
            self.req.advance(target)?;
            return Ok(NO_MORE_DOCS);
        }
        let mut req_doc = target;
        'head: loop {
            if self.min_score != 0.0 {
                req_doc = self.advance_impacts(req_doc)?;
            }
            if self.req.doc_id() < req_doc {
                req_doc = self.req.advance(req_doc)?;
            }
            if req_doc == NO_MORE_DOCS || !self.opt_is_required {
                return Ok(req_doc);
            }
            let upper_bound = if self.req_max_score < self.min_score {
                NO_MORE_DOCS
            } else {
                self.up_to
            };
            if req_doc > upper_bound {
                continue;
            }
            loop {
                let mut opt_doc = self.opt.doc_id();
                if opt_doc < req_doc {
                    opt_doc = self.opt.advance(req_doc)?;
                }
                if opt_doc > upper_bound {
                    req_doc = upper_bound.saturating_add(1);
                    continue 'head;
                }
                if opt_doc != req_doc {
                    req_doc = self.req.advance(opt_doc)?;
                    if req_doc > upper_bound {
                        continue 'head;
                    }
                }
                if req_doc == NO_MORE_DOCS || opt_doc == req_doc {
                    return Ok(req_doc);
                }
            }
        }
    }
}

impl Scorer for ReqOptSumScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.req.doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        if self.top_scores {
            self.advance_internal(self.req.doc_id().saturating_add(1))
        } else {
            self.req.next_doc()
        }
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.top_scores {
            self.advance_internal(target)
        } else {
            self.req.advance(target)
        }
    }

    fn cost(&self) -> i64 {
        self.req.cost()
    }

    fn two_phase(&self) -> bool {
        self.req.two_phase() || self.opt.two_phase()
    }

    fn matches(&mut self) -> Result<bool> {
        if self.req.two_phase() && !self.req.matches()? {
            return Ok(false);
        }
        if self.opt.two_phase() {
            let doc = self.req.doc_id();
            if self.opt_is_required {
                if self.opt.doc_id() != doc {
                    if self.opt.doc_id() < doc {
                        self.opt.advance(doc)?;
                    }
                    if self.opt.doc_id() != doc {
                        return Ok(false);
                    }
                }
                if !self.opt.matches()? {
                    self.opt.next_doc()?;
                    return Ok(false);
                }
            } else if self.opt.doc_id() == doc && !self.opt.matches()? {
                self.opt.next_doc()?;
            }
        }
        Ok(true)
    }

    fn match_cost(&self) -> f32 {
        let mut cost = 1.0;
        if self.req.two_phase() {
            cost += self.req.match_cost();
        }
        if self.opt.two_phase() {
            cost += self.opt.match_cost();
        }
        cost
    }

    fn score(&mut self) -> Result<f32> {
        let cur = self.req.doc_id();
        let mut score = self.req.score()?;
        let mut opt_doc = self.opt.doc_id();
        if opt_doc < cur {
            opt_doc = self.opt.advance(cur)?;
            if self.opt.two_phase() && opt_doc == cur && !self.opt.matches()? {
                opt_doc = self.opt.next_doc()?;
            }
        }
        if opt_doc == cur {
            // `float score += optScorer.score()`: a float addition, not a
            // double sum.
            score += self.opt.score()?;
        }
        Ok(score)
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        let mut up_to = self.req.advance_shallow(target)?;
        let opt_doc = self.opt.doc_id();
        if opt_doc <= target {
            up_to = up_to.min(self.opt.advance_shallow(target)?);
        } else if opt_doc != NO_MORE_DOCS {
            up_to = up_to.min(opt_doc - 1);
        }
        Ok(up_to)
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut max = self.req.max_score(up_to)?;
        if self.opt.doc_id() <= up_to {
            max += self.opt.max_score(up_to)?;
        }
        Ok(max)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.min_score = min;
        if self.req_max_score < min {
            self.opt_is_required = true;
            if self.req_max_score == 0.0 {
                self.opt.set_min_competitive_score(min)?;
            }
        }
        Ok(())
    }
}
