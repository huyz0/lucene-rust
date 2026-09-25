//! `ConjunctionScorer` (over `ConjunctionDISI` and its
//! `ConjunctionTwoPhaseIterator`) and `BlockMaxConjunctionScorer`.

use lucene_util::fixed_bit_set::FixedBitSet;

use super::{BoxScorer, Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::TermLeg;
use crate::Result;

/// `ConjunctionScorer(required, scorers)`: every required scorer must match;
/// the score is the `double` sum of the scoring ones.
pub(crate) struct ConjunctionScorer<'a> {
    /// Every required scorer, cheapest first: `scorers[0]` leads.
    scorers: Vec<BoxScorer<'a>>,
    /// Indices into `scorers` of the scoring ones.
    scoring: Vec<usize>,
    /// Indices into `scorers` of the two-phase ones, cheapest match first.
    two_phase: Vec<usize>,
    match_cost: f32,
    /// Indices past the lead that are advanced to agree with it.
    leap: Vec<usize>,
    /// Indices past the lead checked by membership (see [`Scorer::contains`]).
    bits: Vec<usize>,
}

impl<'a> ConjunctionScorer<'a> {
    /// `required` do not score; `scoring` do. Together at least two.
    pub(crate) fn new(required: Vec<BoxScorer<'a>>, scoring: Vec<BoxScorer<'a>>) -> Self {
        debug_assert!(required.len() + scoring.len() >= 2);
        let mut all: Vec<(BoxScorer<'a>, bool)> = required
            .into_iter()
            .map(|s| (s, false))
            .chain(scoring.into_iter().map(|s| (s, true)))
            .collect();
        // `CollectionUtil.timSort(iterators, by cost)`: stable.
        all.sort_by_key(|(s, _)| s.cost());
        let mut scorers = Vec::with_capacity(all.len());
        let mut scoring_idx = Vec::new();
        for (i, (s, is_scoring)) in all.into_iter().enumerate() {
            if is_scoring {
                scoring_idx.push(i);
            }
            scorers.push(s);
        }
        let mut two_phase: Vec<usize> = (0..scorers.len())
            .filter(|&i| scorers[i].two_phase())
            .collect();
        two_phase.sort_by(|&a, &b| scorers[a].match_cost().total_cmp(&scorers[b].match_cost()));
        let match_cost = two_phase.iter().map(|&i| scorers[i].match_cost()).sum();
        // `BitSetConjunctionDISI`: a non-lead, non-scoring iterator with random
        // access is checked by membership rather than advanced.
        let (bits, leap): (Vec<usize>, Vec<usize>) = (1..scorers.len()).partition(|&i| {
            !scoring_idx.contains(&i) && !scorers[i].two_phase() && scorers[i].contains(0).is_some()
        });
        Self {
            scorers,
            scoring: scoring_idx,
            two_phase,
            match_cost,
            leap,
            bits,
        }
    }

    /// `ConjunctionDISI.doNext`: leapfrog until every iterator agrees.
    fn do_next(&mut self, mut doc: i32) -> Result<i32> {
        'head: loop {
            if doc == NO_MORE_DOCS {
                return Ok(doc);
            }
            for k in 0..self.leap.len() {
                let other = &mut self.scorers[self.leap[k]];
                if other.doc_id() < doc {
                    let next = other.advance(doc)?;
                    if next > doc {
                        doc = self.scorers[0].advance(next)?;
                        continue 'head;
                    }
                }
            }
            for k in 0..self.bits.len() {
                let other = &mut self.scorers[self.bits[k]];
                match other.contains(doc) {
                    Some(true) => {}
                    Some(false) => {
                        doc = self.scorers[0].next_doc()?;
                        continue 'head;
                    }
                    // No longer random-access: leapfrog as any other.
                    None => {
                        if other.doc_id() < doc {
                            let next = other.advance(doc)?;
                            if next > doc {
                                doc = self.scorers[0].advance(next)?;
                                continue 'head;
                            }
                        }
                    }
                }
            }
            return Ok(doc);
        }
    }
}

impl Scorer for ConjunctionScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.scorers[0].doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        let doc = self.scorers[0].next_doc()?;
        self.do_next(doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.scorers[0].advance(target)?;
        self.do_next(doc)
    }

    fn cost(&self) -> i64 {
        self.scorers[0].cost()
    }

    fn two_phase(&self) -> bool {
        !self.two_phase.is_empty()
    }

    fn matches(&mut self) -> Result<bool> {
        for &i in &self.two_phase {
            if !self.scorers[i].matches()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn match_cost(&self) -> f32 {
        self.match_cost
    }

    fn score(&mut self) -> Result<f32> {
        let mut sum = 0.0f64;
        for &i in &self.scoring {
            sum += f64::from(self.scorers[i].score()?);
        }
        Ok(sum as f32)
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        if let [only] = self.scoring[..] {
            return self.scorers[only].advance_shallow(target);
        }
        for &i in &self.scoring {
            self.scorers[i].advance_shallow(target)?;
        }
        Ok(NO_MORE_DOCS)
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut sum = 0.0f64;
        for &i in &self.scoring {
            let s = &mut self.scorers[i];
            if s.doc_id() <= up_to {
                sum += f64::from(s.max_score(up_to)?);
            }
        }
        Ok(sum as f32)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if let [only] = self.scoring[..] {
            self.scorers[only].set_min_competitive_score(min)?;
        }
        Ok(())
    }
}

/// `BlockMaxConjunctionScorer`: a conjunction of scoring clauses whose
/// approximation skips every block where the clauses' summed maxima cannot
/// reach the minimum competitive score.
pub(crate) struct BlockMaxConjunctionScorer<'a> {
    /// Cheapest first; `scorers[0]` leads.
    scorers: Vec<BoxScorer<'a>>,
    /// Indices of the two-phase scorers, cheapest match first.
    two_phase: Vec<usize>,
    min_score: f32,
    /// The approximation's current block: its last document and bound.
    up_to: i32,
    block_max: f32,
}

impl<'a> BlockMaxConjunctionScorer<'a> {
    pub(crate) fn new(mut scorers: Vec<BoxScorer<'a>>) -> Result<Self> {
        scorers.sort_by_key(|s| s.cost());
        for s in &mut scorers {
            s.advance_shallow(0)?;
        }
        let mut two_phase: Vec<usize> = (0..scorers.len())
            .filter(|&i| scorers[i].two_phase())
            .collect();
        two_phase.sort_by(|&a, &b| scorers[a].match_cost().total_cmp(&scorers[b].match_cost()));
        Ok(Self {
            scorers,
            two_phase,
            min_score: 0.0,
            up_to: -1,
            block_max: 0.0,
        })
    }

    fn move_to_next_block(&mut self, target: i32) -> Result<()> {
        if self.min_score == 0.0 {
            self.up_to = target;
            self.block_max = f32::INFINITY;
        } else {
            self.up_to = self.advance_shallow(target)?;
            self.block_max = self.max_score(self.up_to)?;
        }
        Ok(())
    }

    fn advance_target(&mut self, mut target: i32) -> Result<i32> {
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

    fn do_next(&mut self, mut doc: i32) -> Result<i32> {
        'head: loop {
            if doc == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            }
            if doc > self.up_to {
                let next_target = self.advance_target(doc)?;
                if next_target != doc {
                    doc = self.scorers[0].advance(next_target)?;
                    continue;
                }
            }
            for i in 1..self.scorers.len() {
                let other = &mut self.scorers[i];
                if other.doc_id() < doc {
                    let next = other.advance(doc)?;
                    if next > doc {
                        let target = self.advance_target(next)?;
                        doc = self.scorers[0].advance(target)?;
                        continue 'head;
                    }
                }
            }
            return Ok(doc);
        }
    }
}

impl Scorer for BlockMaxConjunctionScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.scorers[0].doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc_id().saturating_add(1))
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        let target = self.advance_target(target)?;
        let doc = self.scorers[0].advance(target)?;
        self.do_next(doc)
    }

    fn cost(&self) -> i64 {
        self.scorers[0].cost()
    }

    fn two_phase(&self) -> bool {
        !self.two_phase.is_empty()
    }

    fn matches(&mut self) -> Result<bool> {
        for &i in &self.two_phase {
            if !self.scorers[i].matches()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn match_cost(&self) -> f32 {
        self.two_phase
            .iter()
            .map(|&i| f64::from(self.scorers[i].match_cost()))
            .sum::<f64>() as f32
    }

    fn score(&mut self) -> Result<f32> {
        let mut sum = 0.0f64;
        for s in &mut self.scorers {
            sum += f64::from(s.score()?);
        }
        Ok(sum as f32)
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        let result = self.scorers[0].advance_shallow(target)?;
        for s in &mut self.scorers[1..] {
            s.advance_shallow(target)?;
        }
        Ok(result)
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut sum = 0.0f64;
        for s in &mut self.scorers {
            sum += f64::from(s.max_score(up_to)?);
        }
        Ok(sum as f32)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.min_score = min;
        Ok(())
    }
}

/// [`ConjunctionScorer`] over term legs only: the same leapfrog and the same
/// `double` sum, with every clause call dispatched statically. Built when
/// every required clause is a term, a boosted term or a constant-scored term,
/// which is most conjunctions nested in a scorer tree.
pub(crate) struct LegConjunctionScorer<'a> {
    /// Cheapest first; `legs[0]` leads.
    legs: Vec<TermLeg<'a>>,
    /// Indices into `legs` of the scoring ones.
    scoring: Vec<usize>,
    /// `TOP_SCORES`: a lone scoring leg may be handed a threshold, after
    /// which it iterates through its impacts (`ImpactsDISI`).
    top_scores: bool,
    impacts: bool,
}

impl<'a> LegConjunctionScorer<'a> {
    /// `required` do not score; `scoring` do. Together at least two.
    pub(crate) fn new(
        required: Vec<TermLeg<'a>>,
        scoring: Vec<TermLeg<'a>>,
        top_scores: bool,
    ) -> Self {
        let mut all: Vec<(TermLeg<'a>, bool)> = required
            .into_iter()
            .map(|l| (l, false))
            .chain(scoring.into_iter().map(|l| (l, true)))
            .collect();
        all.sort_by_key(|(l, _)| l.cost);
        let mut legs = Vec::with_capacity(all.len());
        let mut scoring_idx = Vec::new();
        for (i, (l, is_scoring)) in all.into_iter().enumerate() {
            if is_scoring {
                scoring_idx.push(i);
            }
            legs.push(l);
        }
        Self {
            legs,
            scoring: scoring_idx,
            top_scores,
            impacts: false,
        }
    }

    #[inline]
    fn advance_leg(&mut self, i: usize, target: i32) -> Result<i32> {
        if self.impacts && self.scoring[0] == i {
            self.legs[i].impacts_advance(target)
        } else {
            self.legs[i].advance(target)
        }
    }

    fn do_next(&mut self, mut doc: i32) -> Result<i32> {
        'head: loop {
            if doc == NO_MORE_DOCS {
                return Ok(doc);
            }
            for i in 1..self.legs.len() {
                if self.legs[i].doc_id() < doc {
                    let next = self.advance_leg(i, doc)?;
                    if next > doc {
                        doc = self.advance_leg(0, next)?;
                        continue 'head;
                    }
                }
            }
            return Ok(doc);
        }
    }
}

impl Scorer for LegConjunctionScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.legs[0].doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        let doc = if self.impacts && self.scoring[0] == 0 {
            self.legs[0].impacts_next_doc()?
        } else {
            self.legs[0].next_doc()?
        };
        self.do_next(doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.advance_leg(0, target)?;
        self.do_next(doc)
    }

    fn cost(&self) -> i64 {
        self.legs[0].cost
    }

    fn score(&mut self) -> Result<f32> {
        let mut sum = 0.0f64;
        for &i in &self.scoring {
            sum += f64::from(self.legs[i].score()?);
        }
        Ok(sum as f32)
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        if let [only] = self.scoring[..] {
            return self.legs[only].shallow_advance(target);
        }
        for &i in &self.scoring {
            self.legs[i].shallow_advance(target)?;
        }
        Ok(NO_MORE_DOCS)
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut sum = 0.0f64;
        for &i in &self.scoring {
            let leg = &mut self.legs[i];
            if leg.doc_id() <= up_to {
                sum += f64::from(leg.max_score(up_to));
            }
        }
        Ok(sum as f32)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if let [only] = self.scoring[..] {
            if self.top_scores && min > 0.0 {
                self.legs[only].set_min_competitive_score(min);
                self.impacts = true;
            }
        }
        Ok(())
    }

    /// The default, with the leapfrog and the scoring inlined: a nested
    /// conjunction a disjunction iterates as an essential clause spends its
    /// time here.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        out.docs.clear();
        out.scores.clear();
        let mut doc = self.legs[0].doc_id();
        while doc < up_to && out.docs.len() < super::NEXT_DOCS_BATCH {
            if live_docs.is_none_or(|l| l.get_doc(doc)) {
                out.docs.push(doc);
                out.scores.push(Scorer::score(self)?);
            }
            doc = Scorer::next_doc(self)?;
        }
        Ok(())
    }
}
