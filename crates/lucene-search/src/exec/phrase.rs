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
//! block-max bound is the similarity's global one (`weight`) rather than
//! `ExactPhraseMatcher`'s merged impacts, and `matches`' `maxFreq` check runs
//! for an exact phrase only (the sloppy matcher's `maxFreq` is not ported).
//! Both only mean less pruning.

use lucene_codecs::postings::PositionsCursor;

use super::{Scorer, NO_MORE_DOCS};
use crate::field_norms::FieldNormsCursor;
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
    /// Per slot, for the first-occurrence check: the current position minus
    /// the slot, and the occurrences not yet read.
    lazy: Vec<(i32, i32)>,
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
            norms,
            norm_doc: -1,
            norm_inverse: similarity::UNNORMED_NORM_INVERSE,
            freq: 0.0,
            top_scores,
            needs_scores,
            min_competitive: 0.0,
            match_cost,
            lazy: vec![(0, 0); n],
        }
    }

    /// `ExactPhraseMatcher.nextMatch` from the start of the document: whether
    /// the phrase occurs at all, reading each term's positions only as far as
    /// the search needs. The rest are left for the cursor to skip.
    fn exact_occurs(&mut self) -> Result<bool> {
        let pe = |e| -> crate::Error { blocktree::Error::Postings(e).into() };
        for t in self.terms.iter_mut() {
            let first = t.cursor.next_position().map_err(pe)?;
            self.lazy[t.slot] = (first - t.slot as i32, t.cursor.freq() - 1);
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
                    *rel = t.cursor.next_position().map_err(pe)? - t.slot as i32;
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
        let doc = self.terms[0]
            .cursor
            .next_doc()
            .map_err(|e| crate::Error::from(blocktree::Error::Postings(e)))?;
        self.do_next(doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
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
        if self.top_scores && self.min_competitive > 0.0 && self.slop == 0 {
            // `matcher.maxFreq()`: the phrase occurs at most as often as its
            // rarest term, so a document that cannot compete even then is
            // rejected before any position is read.
            let max_freq = self
                .terms
                .iter()
                .map(|t| t.cursor.freq())
                .min()
                .unwrap_or(0) as f32;
            let norm_inverse = self.norm_inverse(doc)?;
            if similarity::do_score(self.weight, max_freq, norm_inverse) < self.min_competitive {
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
            for _ in 0..t.cursor.freq() {
                buf.push(t.cursor.next_position().map_err(pe)?);
            }
        }
        let slices: Vec<&[i32]> = self.positions.iter().map(Vec::as_slice).collect();
        self.freq = if self.slop == 0 {
            crate::phrase_freq_exact(&slices) as f32
        } else {
            sloppy_phrase::sloppy_phrase_freq(&slices, &self.repeats, self.slop)
        };
        Ok(self.freq > 0.0)
    }

    fn match_cost(&self) -> f32 {
        self.match_cost
    }

    fn score(&mut self) -> Result<f32> {
        let doc = self.doc_id();
        let norm_inverse = self.norm_inverse(doc)?;
        Ok(similarity::do_score(self.weight, self.freq, norm_inverse))
    }

    fn advance_shallow(&mut self, _target: i32) -> Result<i32> {
        Ok(NO_MORE_DOCS)
    }

    /// BM25's `weight - weight / (1 + freq * normInverse)` never reaches
    /// `weight`.
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(self.weight)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.min_competitive = min;
        Ok(())
    }
}
