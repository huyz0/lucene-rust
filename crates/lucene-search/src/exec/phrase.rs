//! `PhraseScorer`: a phrase as a two-phase scorer, so it composes with every
//! other scorer in the tree (a `bool` clause, a `dis_max` disjunct, a
//! `constant_score` body) instead of being resolved up front.
//!
//! The approximation is the conjunction of the terms' postings, rarest term
//! leading (`ConjunctionUtils.intersectIterators`); [`Scorer::matches`] reads
//! the candidate's positions and counts the phrase with the same functions the
//! lone-phrase path uses ([`crate::phrase_freq_exact`],
//! [`sloppy_phrase::sloppy_phrase_freq`]), so a document scores the same bits
//! either way.
//!
//! Deviations from Lucene, neither of which changes a hit or a score: the
//! block-max bound is the similarity's global one (`weight` for BM25,
//! `score(Float.MAX_VALUE, 1)` for any other) rather than
//! `ExactPhraseMatcher`'s merged impacts, which only means less pruning.
//!
//! A similarity other than the default BM25 scores through its
//! [`SimScorer`] (`PhraseScorer.score`'s `simScorer.score(freq, norm)`, the
//! norm as stored, `1` without one); BM25 keeps its weight and norm-inverse
//! table.

use lucene_codecs::postings::PositionsCursor;

use std::sync::Arc;

use super::{Scorer, NO_MORE_DOCS};
use crate::field_norms::FieldNormsCursor;
use crate::similarities::SimScorer;
use crate::{blocktree, similarity, sloppy_phrase, Result};

/// `PhraseQuery.TERM_POSNS_SEEK_OPS_PER_DOC`.
const TERM_POSNS_SEEK_OPS_PER_DOC: f32 = 256.0;
/// `PhraseQuery.TERM_OPS_PER_POS`.
const TERM_OPS_PER_POS: f32 = 7.0;

/// `PhraseQuery.termPositionsCost`: the expected cost of reading one matching
/// document's positions for a term.
pub(crate) fn term_positions_cost(doc_freq: i64, total_term_freq: i64) -> f32 {
    let exp_occurrences = total_term_freq as f32 / doc_freq.max(1) as f32;
    TERM_POSNS_SEEK_OPS_PER_DOC + exp_occurrences * TERM_OPS_PER_POS
}

/// One term of the phrase: its cursor and its slot in the phrase's order.
pub(crate) struct PhraseTerm<'a> {
    pub(crate) cursor: PositionsCursor<'a>,
    pub(crate) slot: usize,
    pub(crate) cost: i64,
}

pub(crate) struct PhraseScorer<'a> {
    /// Cheapest first; `terms[0]` leads the conjunction.
    terms: Vec<PhraseTerm<'a>>,
    positions: Vec<Vec<i32>>,
    weight: f32,
    slop: u32,
    repeats: sloppy_phrase::PhraseRepeats,
    /// The sloppy matcher's buffers, reused across documents.
    scratch: sloppy_phrase::SloppyScratch,
    norms: Option<FieldNormsCursor<'a, 'a>>,
    /// `norms` read for `norm_doc`, so `matches` and `score` read it once.
    norm_doc: i32,
    norm_inverse: f32,
    /// The phrase frequency `matches` counted for the current document.
    freq: f32,
    top_scores: bool,
    /// Scores are read, so `matches` counts the whole frequency; otherwise it
    /// stops at the first occurrence, as `ExactPhraseMatcher.nextMatch` does.
    needs_scores: bool,
    min_competitive: f32,
    match_cost: f32,
    /// A similarity other than the default BM25, and its global bound.
    sim: Option<(Arc<dyn SimScorer>, f32)>,
    /// Per slot, for the first-occurrence check: the current position minus
    /// the slot, and the occurrences not yet read.
    lazy: Vec<(i32, i32)>,
    /// `PhraseQuery.getPositions()` when they are not the implicit `0..n`
    /// (`Builder.add(term, position)`), rebased so the first is `0`.
    offsets: Option<Vec<i32>>,
    /// Scratch for rebasing positions onto slots.
    shifted: Vec<Vec<i32>>,
}

impl<'a> PhraseScorer<'a> {
    /// `terms` in any order; `weight` is `boost * sum(idf)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mut terms: Vec<PhraseTerm<'a>>,
        weight: f32,
        slop: u32,
        repeats: sloppy_phrase::PhraseRepeats,
        norms: Option<FieldNormsCursor<'a, 'a>>,
        match_cost: f32,
        top_scores: bool,
        needs_scores: bool,
    ) -> Self {
        terms.sort_by_key(|t| t.cost);
        let n = terms.len();
        Self {
            terms,
            positions: vec![Vec::new(); n],
            weight,
            slop,
            repeats,
            scratch: sloppy_phrase::SloppyScratch::default(),
            norms,
            norm_doc: -1,
            norm_inverse: similarity::UNNORMED_NORM_INVERSE,
            freq: 0.0,
            top_scores,
            needs_scores,
            min_competitive: 0.0,
            match_cost,
            sim: None,
            lazy: vec![(0, 0); n],
            offsets: None,
            shifted: Vec::new(),
        }
    }

    /// Explicit per-slot positions (`PhraseQuery.Builder.add(term,
    /// position)`), one per slot in phrase order.
    pub(crate) fn with_offsets(mut self, offsets: Vec<i32>) -> Self {
        let identity = offsets
            .iter()
            .enumerate()
            .all(|(i, &o)| i64::from(o) == i as i64);
        if !identity {
            self.offsets = Some(offsets);
        }
        self
    }

    /// Slot `slot`'s offset in the phrase.
    fn offset(&self, slot: usize) -> i32 {
        match &self.offsets {
            Some(o) => o[slot],
            None => slot as i32,
        }
    }

    /// Scores through `scorer` instead of BM25's weight: a phrase under a
    /// similarity other than the default.
    pub(crate) fn with_sim_scorer(mut self, scorer: Arc<dyn SimScorer>) -> Self {
        let max = crate::bulk_scorer::sim_global_max(scorer.as_ref(), f32::MAX);
        self.sim = Some((scorer, max));
        self
    }

    /// `simScorer.score(freq, norm)` for the current document, under a
    /// similarity other than the default BM25.
    fn sim_score(&mut self, freq: f32) -> Result<f32> {
        let doc = self.doc_id();
        let norm = match self.norms.as_mut() {
            Some(n) => n.norm_long(doc)?.unwrap_or(1),
            None => 1,
        };
        Ok(self.sim.as_ref().map_or(0.0, |(s, _)| s.score(freq, norm)))
    }

    /// `ExactPhraseMatcher.nextMatch` from the start of the document: whether
    /// the phrase occurs at all, reading each term's positions only as far as
    /// the search needs. The rest are left for the cursor to skip.
    fn exact_occurs(&mut self) -> Result<bool> {
        let pe = |e| -> crate::Error { blocktree::Error::Postings(e).into() };
        let offsets: Vec<i32> = (0..self.terms.len()).map(|s| self.offset(s)).collect();
        for t in self.terms.iter_mut() {
            let first = t.cursor.next_position().map_err(pe)?;
            self.lazy[t.slot] = (first - offsets[t.slot], t.cursor.freq() - 1);
        }
        loop {
            // Every term must sit at the same phrase start.
            let target = self.lazy.iter().map(|&(rel, _)| rel).max().unwrap_or(0);
            let mut aligned = true;
            for t in self.terms.iter_mut() {
                let (rel, left) = &mut self.lazy[t.slot];
                while *rel < target {
                    if *left == 0 {
                        return Ok(false);
                    }
                    *left -= 1;
                    *rel = t.cursor.next_position().map_err(pe)? - offsets[t.slot];
                }
                aligned &= *rel == target;
            }
            if aligned {
                return Ok(true);
            }
        }
    }

    fn do_next(&mut self, mut doc: i32) -> Result<i32> {
        let pe = |e| -> crate::Error { blocktree::Error::Postings(e).into() };
        'head: loop {
            if doc == NO_MORE_DOCS {
                return Ok(doc);
            }
            for i in 1..self.terms.len() {
                let mut other = self.terms[i].cursor.doc_id();
                if other < doc {
                    other = self.terms[i].cursor.advance(doc).map_err(pe)?;
                }
                if other != doc {
                    doc = self.terms[0].cursor.advance(other).map_err(pe)?;
                    continue 'head;
                }
            }
            return Ok(doc);
        }
    }

    /// `ImpactsDISI` once the threshold passes the bound: no document of
    /// this phrase can score above [`Scorer::max_score`], so none reaches
    /// `setMinCompetitiveScore`'s value (Java's own, `Math.nextUp` included
    /// when ties lose) and the iterator ends.
    fn cannot_compete(&self) -> bool {
        self.top_scores
            && self.min_competitive > 0.0
            && match &self.sim {
                None => self.weight,
                Some((_, max)) => *max,
            } < self.min_competitive
    }

    /// Moves the lead cursor past the last document: `NO_MORE_DOCS`.
    fn exhaust(&mut self) -> Result<i32> {
        self.terms[0]
            .cursor
            .advance(NO_MORE_DOCS)
            .map_err(|e| crate::Error::from(blocktree::Error::Postings(e)))
    }

    fn norm_inverse(&mut self, doc: i32) -> Result<f32> {
        if self.norm_doc != doc {
            self.norm_inverse = match self.norms.as_mut() {
                Some(n) => n.norm_inverse(doc)?,
                None => similarity::UNNORMED_NORM_INVERSE,
            };
            self.norm_doc = doc;
        }
        Ok(self.norm_inverse)
    }
}

impl Scorer for PhraseScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.terms[0].cursor.doc_id()
    }

    fn next_doc(&mut self) -> Result<i32> {
        if self.cannot_compete() {
            return self.exhaust();
        }
        let doc = self.terms[0]
            .cursor
            .next_doc()
            .map_err(|e| crate::Error::from(blocktree::Error::Postings(e)))?;
        self.do_next(doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.cannot_compete() {
            return self.exhaust();
        }
        let doc = self.terms[0]
            .cursor
            .advance(target)
            .map_err(|e| crate::Error::from(blocktree::Error::Postings(e)))?;
        self.do_next(doc)
    }

    fn cost(&self) -> i64 {
        self.terms[0].cost
    }

    fn two_phase(&self) -> bool {
        true
    }

    /// `PhraseScorer.twoPhaseIterator().matches()`.
    fn matches(&mut self) -> Result<bool> {
        let doc = self.doc_id();
        if self.top_scores && self.min_competitive > 0.0 {
            // `matcher.maxFreq()`, so a document that cannot compete even at
            // that frequency is rejected before any position is read. Exact:
            // the phrase occurs at most as often as its rarest term. Sloppy:
            // each term position heads at most one match, each weighing at
            // most 1, so at most the sum of the frequencies (a `float` sum,
            // as `SloppyPhraseMatcher.maxFreq` adds them).
            let max_freq = if self.slop == 0 {
                self.terms
                    .iter()
                    .map(|t| t.cursor.freq())
                    .min()
                    .unwrap_or(0) as f32
            } else {
                self.terms
                    .iter()
                    .fold(0.0f32, |sum, t| sum + t.cursor.freq() as f32)
            };
            let bound = if self.sim.is_none() {
                let norm_inverse = self.norm_inverse(doc)?;
                similarity::do_score(self.weight, max_freq, norm_inverse)
            } else {
                self.sim_score(max_freq)?
            };
            if bound < self.min_competitive {
                return Ok(false);
            }
        }
        if !self.needs_scores && self.slop == 0 {
            return self.exact_occurs();
        }
        let pe = |e| -> crate::Error { blocktree::Error::Postings(e).into() };
        for t in self.terms.iter_mut() {
            let buf = &mut self.positions[t.slot];
            buf.clear();
            t.cursor.positions_into(buf).map_err(pe)?;
        }
        if let Some(offsets) = &self.offsets {
            self.freq = super::extended::phrase_freq_at(
                &self.positions,
                offsets,
                &self.repeats,
                self.slop,
                &mut self.scratch,
                &mut self.shifted,
            );
            return Ok(self.freq > 0.0);
        }
        // A stack array for any ordinary phrase, a `Vec` only past eight terms.
        let mut inline: [&[i32]; 8] = [&[]; 8];
        let spilled: Vec<&[i32]>;
        let slices: &[&[i32]] = if self.positions.len() <= inline.len() {
            for (slot, p) in inline.iter_mut().zip(&self.positions) {
                *slot = p.as_slice();
            }
            &inline[..self.positions.len()]
        } else {
            spilled = self.positions.iter().map(Vec::as_slice).collect();
            &spilled
        };
        self.freq = if self.slop == 0 {
            crate::phrase_freq_exact(slices) as f32
        } else {
            sloppy_phrase::sloppy_phrase_freq_in(
                &mut self.scratch,
                slices,
                &self.repeats,
                self.slop,
            )
        };
        Ok(self.freq > 0.0)
    }

    fn match_cost(&self) -> f32 {
        self.match_cost
    }

    fn score(&mut self) -> Result<f32> {
        if self.sim.is_some() {
            return self.sim_score(self.freq);
        }
        let doc = self.doc_id();
        let norm_inverse = self.norm_inverse(doc)?;
        Ok(similarity::do_score(self.weight, self.freq, norm_inverse))
    }

    fn advance_shallow(&mut self, _target: i32) -> Result<i32> {
        Ok(NO_MORE_DOCS)
    }

    /// BM25's `weight - weight / (1 + freq * normInverse)` never reaches
    /// `weight`; any other similarity's `MaxScoreCache.globalMaxScore`.
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(match &self.sim {
            None => self.weight,
            Some((_, max)) => *max,
        })
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.min_competitive = min;
        Ok(())
    }
}
