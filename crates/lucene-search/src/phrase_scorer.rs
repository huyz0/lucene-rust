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
    let mut legs: Vec<(PhraseLeg<'_>, i64)> = cursors
        .into_iter()
        .enumerate()
        .map(|(slot, (cursor, cost))| (PhraseLeg { cursor, slot }, cost))
        .collect();
    // `ConjunctionDISI` leads with the cheapest iterator.
    legs.sort_by_key(|(_, cost)| *cost);
    let mut legs: Vec<PhraseLeg<'_>> = legs.into_iter().map(|(l, _)| l).collect();

    let mut positions: Vec<Vec<i32>> = vec![Vec::new(); n];
    // The sloppy matcher's buffers, reused across documents.
    let mut scratch = sloppy_phrase::SloppyScratch::default();
    let unnormed = similarity::UNNORMED_NORM_INVERSE;

    let pe = |e| -> crate::Error { blocktree::Error::Postings(e).into() };
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
                similarity::do_score(weight, max_freq, norm_inverse) >= min_competitive
            };
            if competitive {
                for leg in legs.iter_mut() {
                    let buf = &mut positions[leg.slot];
                    buf.clear();
                    for _ in 0..leg.cursor.freq() {
                        buf.push(leg.cursor.next_position().map_err(pe)?);
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::TopDocsCollector;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::postings::{DocInput, PosInput};
    use lucene_codecs::postings_writer::{write_single_field, FieldPostingsInput, TermPostings};

    const SEG_ID: [u8; 16] = [11u8; 16];

    /// Keeps every hit, and asks for no pruning.
    #[derive(Default)]
    struct AllHits(Vec<(i32, f32)>);
    impl ScoringCollector for AllHits {
        fn collect(&mut self, doc_id: i32, score: f32) {
            self.0.push((doc_id, score));
        }
    }

    /// `score_phrase` over freshly written postings agrees with scoring every
    /// document by hand from its token list: the same documents with the
    /// same scores for exact and sloppy phrases, for a phrase longer than the
    /// eight-slice stack array, and -- through a small top-k collector, whose
    /// threshold drives the `maxFreq` pre-check -- the same top hits.
    #[test]
    fn score_phrase_agrees_with_a_brute_force_count() {
        let vocab = 4usize; // few terms, so phrases match often
        let mut x = 0x5EED_1234_ABCD_9876u64;
        let docs: Vec<Vec<usize>> = (0..600)
            .map(|_| {
                (0..20)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        (x % vocab as u64) as usize
                    })
                    .collect()
            })
            .collect();
        let names: Vec<Vec<u8>> = (0..vocab).map(|t| format!("w{t}").into_bytes()).collect();
        let terms: Vec<TermPostings> = (0..vocab)
            .map(|t| {
                let mut tp = TermPostings {
                    term: names[t].clone(),
                    ..Default::default()
                };
                for (d, toks) in docs.iter().enumerate() {
                    let pos: Vec<i32> = toks
                        .iter()
                        .enumerate()
                        .filter(|(_, &w)| w == t)
                        .map(|(p, _)| p as i32)
                        .collect();
                    if !pos.is_empty() {
                        tp.docs.push((d as i32, pos.len() as i32));
                        tp.positions.push(pos);
                    }
                }
                tp
            })
            .collect();
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: docs.len() as i32,
            has_payloads: false,
            terms: &terms,
        };
        let out = write_single_field(&input, &SEG_ID, "").unwrap();
        let fis = FieldInfos {
            fields: vec![FieldInfo {
                name: "body".into(),
                number: 0,
                store_term_vectors: false,
                omit_norms: true,
                store_payloads: false,
                soft_deletes_field: false,
                parent_field: false,
                index_options: IndexOptions::DocsAndFreqsAndPositions,
                doc_values_type: DocValuesType::None,
                doc_values_skip_index_type: DocValuesSkipIndexType::None,
                doc_values_gen: -1,
                attributes: Vec::new(),
                point_dimension_count: 0,
                point_index_dimension_count: 0,
                point_num_bytes: 0,
                vector_dimension: 0,
                vector_encoding: VectorEncoding::Float32,
                vector_similarity_function: VectorSimilarityFunction::Euclidean,
            }],
        };
        let fields = blocktree::open(
            &out.tim,
            &out.tip,
            &out.tmd,
            &fis,
            &SEG_ID,
            "",
            docs.len() as i32,
        )
        .unwrap();
        let field = fields.field("body").unwrap();
        let doc_in = DocInput::open(&out.doc, &SEG_ID, "").unwrap();
        let pos_in = PosInput::open(&out.pos, &SEG_ID, "").unwrap();
        let weight = 1.5f32;

        let phrases: Vec<(Vec<usize>, u32)> = vec![
            (vec![0, 1], 0),
            (vec![2, 2], 0),
            (vec![3, 1, 0], 1),
            (vec![0, 1, 2, 3, 0, 1, 2, 3, 0], 0), // nine terms: past the stack array
            (vec![1, 3], 2),
        ];
        for (phrase, slop) in phrases {
            let phrase_terms: Vec<Vec<u8>> = phrase.iter().map(|&t| names[t].clone()).collect();
            let repeats = sloppy_phrase::PhraseRepeats::for_phrase(&phrase_terms);
            let open = || {
                phrase_terms
                    .iter()
                    .map(|t| {
                        let c = field.lazy_positions(t, &doc_in, &pos_in).unwrap().unwrap();
                        let df = field.seek_exact(t).unwrap().doc_freq as i64;
                        (c, df)
                    })
                    .collect::<Vec<_>>()
            };
            // Brute force: each document's positions per phrase slot, then
            // the same frequency functions the scorer uses.
            let mut want: Vec<(i32, f32)> = Vec::new();
            for (d, toks) in docs.iter().enumerate() {
                let per_slot: Vec<Vec<i32>> = phrase
                    .iter()
                    .map(|&t| {
                        toks.iter()
                            .enumerate()
                            .filter(|(_, &w)| w == t)
                            .map(|(p, _)| p as i32)
                            .collect()
                    })
                    .collect();
                if per_slot.iter().any(|p| p.is_empty()) {
                    continue;
                }
                let slices: Vec<&[i32]> = per_slot.iter().map(|p| p.as_slice()).collect();
                let freq = if slop == 0 {
                    crate::phrase_freq_exact(&slices) as f32
                } else {
                    sloppy_phrase::sloppy_phrase_freq(&slices, &repeats, slop)
                };
                if freq > 0.0 {
                    let norm = similarity::UNNORMED_NORM_INVERSE;
                    want.push((d as i32, similarity::do_score(weight, freq, norm)));
                }
            }
            let mut all = AllHits::default();
            score_phrase(open(), weight, slop, &repeats, None, None, &mut all).unwrap();
            assert_eq!(all.0, want, "phrase {phrase:?} slop {slop}");

            // A threshold of one hit, so pruning (and the `maxFreq` check)
            // starts as soon as five hits are held.
            let mut top = TopDocsCollector::with_total_hits_threshold(5, 1);
            score_phrase(open(), weight, slop, &repeats, None, None, &mut top).unwrap();
            let mut ranked = want.clone();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            ranked.truncate(5);
            let got: Vec<(i32, f32)> = top.top_docs().iter().map(|h| (h.doc_id, h.score)).collect();
            assert_eq!(got, ranked, "top-5, phrase {phrase:?} slop {slop}");
        }
    }
}
