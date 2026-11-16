#![cfg(test)]
use super::*;
use crate::collector::TopDocsCollector;

/// The threshold answers exactly what computing the score does, across
/// weights, minima and `x` values on both sides of it -- including its
/// own bits and their neighbours.
#[test]
fn min_competitive_x_is_the_score_comparison() {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let unit = |v: u64| (v >> 40) as f32 / (1u64 << 24) as f32;
    for _ in 0..2_000 {
        let weight = unit(next()) * 12.0 + 0.01;
        let min = unit(next()) * weight * 1.1;
        let t = min_competitive_x(weight, min);
        let mut xs: Vec<f32> = (0..20).map(|_| unit(next()) * 50.0).collect();
        if let Some(t) = t {
            xs.extend([t, f32::from_bits(t.to_bits() + 1)]);
            if t > 0.0 {
                xs.push(f32::from_bits(t.to_bits() - 1));
            }
        }
        for x in xs {
            let want = weight - weight / (1.0 + x) >= min;
            assert_eq!(
                t.is_some_and(|t| x >= t),
                want,
                "w {weight} min {min} x {x}"
            );
        }
    }
    assert_eq!(
        min_competitive_x(2.0, 3.0),
        None,
        "above the score's ceiling"
    );
    assert_eq!(min_competitive_x(2.0, 0.0), Some(0.0));
}
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

        // A threshold of one hit (clamped to the five kept), so pruning
        // (and the `maxFreq` check) starts once five hits are held.
        let mut top = TopDocsCollector::with_total_hits_threshold(5, 1);
        score_phrase(open(), weight, slop, &repeats, None, None, &mut top).unwrap();
        let mut ranked = want.clone();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(5);
        let got: Vec<(i32, f32)> = top.top_docs().iter().map(|h| (h.doc_id, h.score)).collect();
        assert_eq!(got, ranked, "top-5, phrase {phrase:?} slop {slop}");
    }
}
