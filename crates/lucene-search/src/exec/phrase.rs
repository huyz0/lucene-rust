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
//! An exact phrase bounds its scores as `ExactPhraseMatcher.mergeImpacts`
//! does: block boundaries are the rarest term's, and a block's impacts are
//! every term's merged by frequency ([`merge_impacts`]). Under `TOP_SCORES` its
//! approximation skips blocks as `ImpactsDISI` does. A sloppy phrase has
//! Lucene's dummy impacts: one level, up to `NO_MORE_DOCS`, bounded by the
//! global maximum. Past the hit threshold these bounds decide which
//! documents a disjunction hands the collector, so they are Lucene's exactly.
//!
//! A similarity other than the default BM25 scores through its
//! [`SimScorer`] (`PhraseScorer.score`'s `simScorer.score(freq, norm)`, the
//! norm as stored, `1` without one); BM25 keeps its weight and norm-inverse
//! table.

use lucene_codecs::postings::Impact;

use std::sync::Arc;

use super::{Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::PhrasePostings;
use crate::field_norms::FieldNormsCursor;
use crate::similarities::SimScorer;
use crate::{similarity, sloppy_phrase, Result};

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
    pub(crate) cursor: PhrasePostings<'a>,
    pub(crate) slot: usize,
    pub(crate) cost: i64,
}

/// Java: `PhraseQuery.PostingsAndFreq.compareTo`, before the conjunction's
/// stable cost sort. Equal-cost terms at the same position must use lexical
/// order so `ExactPhraseMatcher.mergeImpacts` chooses the same lead.
pub(crate) fn sort_exact_phrase_terms(
    terms: &mut [PhraseTerm<'_>],
    query: &crate::query::PhraseQuery,
    positions: &[i32],
) {
    if query.slop == 0 {
        terms.sort_by(|a, b| {
            positions[a.slot]
                .cmp(&positions[b.slot])
                .then_with(|| query.terms[a.slot].cmp(&query.terms[b.slot]))
        });
    }
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
    /// `MaxScoreCache.maxScoreCache` / `maxScoreCacheUpTo`, per lead level.
    max_score_cache: [(i32, f32); 4],
    /// `ImpactsDISI.upTo`: `NO_MORE_DOCS` until a threshold is set.
    impacts_up_to: i32,
    /// Scratch for [`merge_impacts`]: each term's level, copied.
    term_impacts: Vec<Vec<Impact>>,
    merged: Vec<Impact>,
}

/// `Integer.MAX_VALUE, 1L`: the impacts Lucene reports where it knows
/// nothing (the tail block), which bound a block at the global maximum.
const DUMMY_IMPACT: Impact = Impact {
    freq: i32::MAX,
    norm: 1,
};

/// `ExactPhraseMatcher.mergeImpacts(...).getImpacts().getImpacts(level)`'s
/// walk: the terms' lists merged by frequency, each step the lowest
/// frequency still listed with the highest (unsigned) norm seen so far,
/// stopping as soon as any one list runs out. Lists that are empty or
/// dummy were left out by the caller; with none left the result is the
/// dummy impact.
fn merge_impacts(lists: &[&[Impact]], out: &mut Vec<Impact>) {
    out.clear();
    match lists {
        [] => {
            out.push(DUMMY_IMPACT);
            return;
        }
        [only] => {
            out.extend_from_slice(only);
            return;
        }
        _ => {}
    }
    // Every list's next index: an iterator's `freq`/`norm` are
    // `lists[i][at[i] - 1]`, as `SubIterator` reads them after `next()`.
    let mut at: Vec<usize> = vec![1; lists.len()];
    let impact_at = |at: &[usize], i: usize| lists[i][at[i] - 1];
    let mut current_freq = (0..lists.len())
        .map(|i| impact_at(&at, i).freq)
        .min()
        .unwrap_or(i32::MAX);
    let mut current_norm = (0..lists.len())
        .map(|i| impact_at(&at, i).norm as u64)
        .max()
        .unwrap_or(0);
    loop {
        match out.last_mut() {
            Some(last) if last.norm as u64 == current_norm => last.freq = current_freq,
            _ => out.push(Impact {
                freq: current_freq,
                norm: current_norm as i64,
            }),
        }
        // Each list on the current frequency steps once: frequencies are
        // strictly increasing within a list, so none is stepped twice.
        for i in 0..lists.len() {
            if impact_at(&at, i).freq != current_freq {
                continue;
            }
            if at[i] >= lists[i].len() {
                // One clause has no documents left at or below this norm.
                return;
            }
            at[i] += 1;
            current_norm = current_norm.max(impact_at(&at, i).norm as u64);
        }
        current_freq = (0..lists.len())
            .map(|i| impact_at(&at, i).freq)
            .min()
            .unwrap_or(i32::MAX);
    }
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
            max_score_cache: [(-1, 0.0); 4],
            impacts_up_to: NO_MORE_DOCS,
            term_impacts: vec![Vec::new(); n],
            merged: Vec::new(),
        }
    }

    /// `MaxScoreCache.globalMaxScore`: BM25's `weight`, or the similarity's.
    fn global_max(&self) -> f32 {
        match &self.sim {
            None => self.weight,
            Some((_, max)) => *max,
        }
    }

    /// The largest score `impacts` allow, as `MaxScoreCache.computeMaxScore`.
    fn impacts_max(&self, impacts: &[Impact]) -> f32 {
        match &self.sim {
            None => crate::bulk_scorer::bound(
                self.norms
                    .as_ref()
                    .map(FieldNormsCursor::norm_inverse_table),
                self.weight,
                impacts,
            ),
            Some((s, _)) => {
                let with_norms = self.norms.is_some();
                impacts.iter().fold(0.0f32, |max, i| {
                    max.max(s.score(i.freq as f32, if with_norms { i.norm } else { 1 }))
                })
            }
        }
    }

    /// The merged impacts' `numLevels()`: the lead's, or a sloppy phrase's one.
    fn merged_num_levels(&self) -> usize {
        if self.slop == 0 {
            self.terms[0].cursor.num_levels()
        } else {
            1
        }
    }

    /// The merged impacts' `getDocIdUpTo(level)`: the lead's.
    fn merged_doc_id_up_to(&self, level: usize) -> i32 {
        if self.slop == 0 {
            self.terms[0].cursor.doc_id_up_to(level)
        } else {
            NO_MORE_DOCS
        }
    }

    /// `MaxScoreCache.getMaxScoreForLevel(level)`, cached as Lucene caches it:
    /// by the level's `docIdUpTo`, not by where the other terms are.
    fn max_score_for_level(&mut self, level: usize) -> f32 {
        let level_up_to = self.merged_doc_id_up_to(level);
        if self.max_score_cache[level].0 < level_up_to {
            let max = if self.slop == 0 {
                self.merged_level_max(level_up_to)
            } else {
                self.impacts_max(&[DUMMY_IMPACT])
            };
            self.max_score_cache[level] = (level_up_to, max);
        }
        self.max_score_cache[level].1
    }

    /// The bound of `mergeImpacts`' `getImpacts(level)`, for the lead level
    /// ending at `level_up_to`.
    fn merged_level_max(&mut self, level_up_to: i32) -> f32 {
        let mut used = 0;
        for t in self.terms.iter_mut() {
            let c = &mut t.cursor;
            // `getLevel`: the term's first level reaching the lead's bound.
            let Some(term_level) = (0..c.num_levels()).find(|&l| c.doc_id_up_to(l) >= level_up_to)
            else {
                continue;
            };
            let out = &mut self.term_impacts[used];
            c.level_impacts(term_level, out);
            if out.is_empty() || out[0] == DUMMY_IMPACT {
                continue;
            }
            used += 1;
        }
        let lists: Vec<&[Impact]> = self.term_impacts[..used]
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut merged = std::mem::take(&mut self.merged);
        merge_impacts(&lists, &mut merged);
        let max = self.impacts_max(&merged);
        self.merged = merged;
        max
    }

    /// `MaxScoreCache.getSkipUpTo(minScore)`: the last doc of the highest
    /// level bounded under `min_score`, or `-1` if level 0 is not.
    //
    // SENTINEL: `-1` = "the current block may compete", as Lucene's; the one
    // caller, `advance_target`, tests `skip == -1`.
    fn skip_up_to(&mut self, min_score: f32) -> i32 {
        let num_levels = self.merged_num_levels();
        let mut level = num_levels - 1;
        for l in 0..num_levels {
            if self.max_score_for_level(l) >= min_score {
                if l == 0 {
                    return -1;
                }
                level = l - 1;
                break;
            }
        }
        self.merged_doc_id_up_to(level)
    }

    /// `ImpactsDISI.advanceTarget` over the merged impacts: the first doc at
    /// or after `target` in a block that can still compete.
    fn advance_target(&mut self, mut target: i32) -> Result<i32> {
        if target <= self.impacts_up_to {
            return Ok(target);
        }
        self.impacts_up_to = self.advance_shallow(target)?;
        let mut max = self.max_score_for_level(0);
        loop {
            if max >= self.min_competitive {
                return Ok(target);
            }
            if self.impacts_up_to == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            }
            let skip = self.skip_up_to(self.min_competitive);
            target = if skip == -1 {
                self.impacts_up_to.saturating_add(1)
            } else if skip == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            } else {
                skip.saturating_add(1)
            };
            self.impacts_up_to = self.advance_shallow(target)?;
            max = self.max_score_for_level(0);
        }
    }

    /// Whether the approximation is `ImpactsDISI`: an exact phrase under
    /// `TOP_SCORES`.
    fn impacts_approximation(&self) -> bool {
        self.top_scores && self.slop == 0
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
        // Split borrows rather than a `Vec` of `self.offset(slot)` per
        // candidate document: this runs once per document a phrase filter
        // checks.
        // Slot `slot`'s offset in the phrase.
        let offsets = self.offsets.as_deref();
        let offset = |slot: usize| offsets.map_or(slot as i32, |o| o[slot]);
        let lazy = &mut self.lazy;
        for t in self.terms.iter_mut() {
            let first = t.cursor.next_position()?;
            lazy[t.slot] = (first - offset(t.slot), t.cursor.freq() - 1);
        }
        loop {
            // Every term must sit at the same phrase start.
            let target = lazy.iter().map(|&(rel, _)| rel).max().unwrap_or(0);
            let mut aligned = true;
            for t in self.terms.iter_mut() {
                let (rel, left) = &mut lazy[t.slot];
                while *rel < target {
                    if *left == 0 {
                        return Ok(false);
                    }
                    *left -= 1;
                    *rel = t.cursor.next_position()? - offset(t.slot);
                }
                aligned &= *rel == target;
            }
            if aligned {
                return Ok(true);
            }
        }
    }

    fn do_next(&mut self, mut doc: i32) -> Result<i32> {
        'head: loop {
            if doc == NO_MORE_DOCS {
                return Ok(doc);
            }
            for i in 1..self.terms.len() {
                let mut other = self.terms[i].cursor.doc_id();
                if other < doc {
                    other = self.terms[i].cursor.advance(doc)?;
                }
                if other != doc {
                    doc = self.terms[0].cursor.advance(other)?;
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
        self.terms[0].cursor.advance(NO_MORE_DOCS)
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
        if self.impacts_approximation() {
            // `ImpactsDISI.nextDoc`.
            let doc = self.doc_id();
            if doc >= self.impacts_up_to {
                return self.advance(doc.saturating_add(1));
            }
        } else if self.cannot_compete() {
            return self.exhaust();
        }
        let doc = self.terms[0].cursor.next_doc()?;
        self.do_next(doc)
    }

    fn advance(&mut self, mut target: i32) -> Result<i32> {
        if self.impacts_approximation() {
            target = self.advance_target(target)?;
        } else if self.cannot_compete() {
            return self.exhaust();
        }
        let doc = self.terms[0].cursor.advance(target)?;
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
        for t in self.terms.iter_mut() {
            let buf = &mut self.positions[t.slot];
            buf.clear();
            t.cursor.positions_into(buf)?;
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

    /// `MaxScoreCache.advanceShallow`: every term moved shallowly, the
    /// lead's block boundary returned. A sloppy phrase has no boundary.
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        if self.slop != 0 {
            return Ok(NO_MORE_DOCS);
        }
        for t in self.terms.iter_mut() {
            t.cursor.advance_shallow(target)?;
        }
        Ok(self.merged_doc_id_up_to(0))
    }

    /// `MaxScoreCache.getMaxScore(upTo)`: the first lead level covering
    /// `up_to`, else the global maximum.
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let level = (0..self.merged_num_levels()).find(|&l| up_to <= self.merged_doc_id_up_to(l));
        Ok(match level {
            Some(l) => self.max_score_for_level(l),
            None => self.global_max(),
        })
    }

    /// `PhraseScorer.setMinCompetitiveScore`, and `ImpactsDISI`'s: a higher
    /// threshold makes the approximation re-check its block.
    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if min > self.min_competitive {
            self.impacts_up_to = -1;
        }
        self.min_competitive = min;
        Ok(())
    }
}
