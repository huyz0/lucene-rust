//! Two-phase phrase scoring over lazy positions: a port of Lucene 10.5.0's
//! `PhraseScorer` driving `ExactPhraseMatcher` / `SloppyPhraseMatcher`.
//!
//! The approximation is the conjunction of the phrase's terms, walked as a
//! leapfrog over [`PositionsCursor`]s (rarest term leads). A candidate
//! document is then confirmed in two steps, the way `PhraseScorer.matches`
//! does it:
//!
//! 1. Once the collector has a threshold, the best score the document could
//!    possibly reach -- its phrase frequency is at most the smallest of its
//!    terms' frequencies (`PhraseMatcher.maxFreq`) -- is checked first. A
//!    document that cannot compete is rejected **without reading a single
//!    position**.
//! 2. Otherwise each term's positions for this document are read into a small
//!    reused buffer and the phrase frequency is counted exactly as the eager
//!    path counted it ([`crate::phrase_freq_exact`] /
//!    [`crate::sloppy_phrase::sloppy_phrase_freq`]), so scores are unchanged.
//!
//! Positions are decoded only for documents that reach step 2, and each
//! term's `.pos` stream is only ever stepped forward.

use lucene_codecs::postings::{PositionsCursor, NO_MORE_DOCS};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::bulk_scorer::min_competitive_score;
use crate::collector::ScoringCollector;
use crate::field_norms::FieldNormsCursor;
use crate::{blocktree, similarity, sloppy_phrase, Result};

/// One phrase term's cursor and its slot in the phrase.
struct PhraseLeg<'a> {
    cursor: PositionsCursor<'a>,
    /// Index into the phrase's terms, i.e. into the per-document position
    /// buffers the matchers read.
    slot: usize,
}

/// Scores every document of the segment that contains the phrase.
///
/// `cursors[i]` is the cursor for `query.terms[i]`; `cost[i]` its document
/// frequency. `weight` is the summed idf, `slop` the query's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_phrase<C: ScoringCollector>(
    cursors: Vec<(PositionsCursor<'_>, i64)>,
    weight: f32,
    slop: u32,
    repeats: &sloppy_phrase::PhraseRepeats,
    mut norms: Option<FieldNormsCursor<'_, '_>>,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
) -> Result<()> {
    let n = cursors.len();
    // `ConjunctionDISI` leads with the cheapest iterator. A cursor holds its
    // block buffers inline (kilobytes), so the order is found on the costs
    // alone (a stable sort, ties in term order) and the cursors are swapped
    // into it in place; the legs then reuse the cursors' own allocation.
    let mut cursors = cursors;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| cursors[i].1);
    // `at[slot]`: where the cursor of term `slot` stands now.
    let mut at: Vec<usize> = (0..n).collect();
    let mut slot_at: Vec<usize> = (0..n).collect();
    for (pos, &slot) in order.iter().enumerate() {
        let from = at[slot];
        if from != pos {
            cursors.swap(pos, from);
            let displaced = slot_at[pos];
            at[displaced] = from;
            slot_at[from] = displaced;
            at[slot] = pos;
            slot_at[pos] = slot;
        }
    }
    for (entry, &slot) in cursors.iter_mut().zip(&order) {
        entry.1 = slot as i64;
    }
    let mut legs: Vec<PhraseLeg<'_>> = cursors
        .into_iter()
        .map(|(cursor, slot)| PhraseLeg {
            cursor,
            slot: slot as usize,
        })
        .collect();

    let mut positions: Vec<Vec<i32>> = vec![Vec::new(); n];
    // The sloppy matcher's buffers, reused across documents.
    let mut scratch = sloppy_phrase::SloppyScratch::default();
    let unnormed = similarity::UNNORMED_NORM_INVERSE;

    let pe = |e| -> crate::Error { blocktree::Error::Postings(e).into() };
    // The collector's threshold the cached `min_x` was found for.
    let mut min_x_for = f32::NAN.to_bits();
    let mut min_x = None;
    let mut doc = legs[0].cursor.next_doc().map_err(pe)?;
    'outer: while doc != NO_MORE_DOCS {
        for i in 1..n {
            let mut other = legs[i].cursor.doc_id();
            if other < doc {
                other = legs[i].cursor.advance(doc).map_err(pe)?;
            }
            if other != doc {
                doc = legs[0].cursor.advance(other).map_err(pe)?;
                continue 'outer;
            }
        }
        if live_docs.is_none_or(|l| l.get_doc(doc)) {
            let norm_inverse = match norms.as_mut() {
                Some(nc) => nc.norm_inverse(doc)?,
                None => unnormed,
            };
            let min_competitive = min_competitive_score(collector);
            if min_competitive.to_bits() != min_x_for {
                min_x_for = min_competitive.to_bits();
                min_x = min_competitive_x(weight, min_competitive);
            }
            // `PhraseScorer.matches`: if even `matcher.maxFreq()` cannot
            // compete, no position needs reading. An exact phrase occurs at
            // most as often as its rarest term; a sloppy one's frequency is at
            // most the `float` sum of its terms' (`SloppyPhraseMatcher.maxFreq`:
            // each position heads at most one match of weight at most 1).
            let competitive = min_competitive <= 0.0 || {
                let max_freq = if slop == 0 {
                    legs.iter().map(|l| l.cursor.freq()).min().unwrap_or(0) as f32
                } else {
                    legs.iter()
                        .fold(0.0f32, |sum, l| sum + l.cursor.freq() as f32)
                };
                // `do_score(weight, max_freq, norm_inverse) >= min_competitive`,
                // without its divide: see `min_competitive_x`.
                min_x.is_some_and(|t| max_freq * norm_inverse >= t)
            };
            if competitive {
                for leg in legs.iter_mut() {
                    let buf = &mut positions[leg.slot];
                    buf.clear();
                    // The document's `freq` positions, checked once per block.
                    leg.cursor.positions_into(buf).map_err(pe)?;
                }
                // Borrowed per candidate: a stack array for any ordinary phrase, a
                // heap `Vec` only past eight terms.
                let mut inline: [&[i32]; 8] = [&[]; 8];
                let spilled: Vec<&[i32]>;
                let slices: &[&[i32]] = if n <= inline.len() {
                    for (slot, p) in inline.iter_mut().zip(&positions) {
                        *slot = p.as_slice();
                    }
                    &inline[..n]
                } else {
                    spilled = positions.iter().map(|p| p.as_slice()).collect();
                    &spilled
                };
                let freq = if slop == 0 {
                    crate::phrase_freq_exact(slices) as f32
                } else {
                    sloppy_phrase::sloppy_phrase_freq_in(&mut scratch, slices, repeats, slop)
                };
                if freq > 0.0 {
                    let score = similarity::do_score(weight, freq, norm_inverse);
                    collector.collect(doc, score);
                }
            }
        }
        doc = legs[0].cursor.next_doc().map_err(pe)?;
    }
    Ok(())
}

/// The smallest `x >= 0` with `weight - weight / (1 + x) >= min` --
/// [`similarity::do_score`] with `x = freq * norm_inverse` -- or `None` when
/// no `x` reaches `min`.
///
/// Every step of that expression rounds monotonically in `x` (the add, the
/// divide and the subtract), so whether a document can compete is exactly
/// whether its `x` is at least this threshold. It only changes when the
/// collector's minimum does, so the binary search over the floats' bit
/// patterns (ordered like the values for `x >= 0`) is paid per threshold
/// rather than a divide per candidate document.
fn min_competitive_x(weight: f32, min: f32) -> Option<f32> {
    let score = |x: f32| weight - weight / (1.0 + x);
    if score(0.0) >= min {
        return Some(0.0);
    }
    let (mut lo, mut hi) = (0.0f32.to_bits(), f32::INFINITY.to_bits());
    if score(f32::from_bits(hi)) < min {
        return None;
    }
    // `score(lo) < min <= score(hi)`.
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if score(f32::from_bits(mid)) >= min {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(f32::from_bits(hi))
}

#[cfg(test)]
mod tests;
