#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::QuantizerFactory;

const ENCODINGS: [ScalarEncoding; 5] = [
    ScalarEncoding::UnsignedByte,
    ScalarEncoding::PackedNibble,
    ScalarEncoding::SevenBit,
    ScalarEncoding::SingleBitQueryNibble,
    ScalarEncoding::DibitQueryNibble,
];
const SIMS: [Similarity; 4] = [
    Similarity::Euclidean,
    Similarity::DotProduct,
    Similarity::Cosine,
    Similarity::MaximumInnerProduct,
];

fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect()
        })
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

/// Estimated scores track exact ones: high correlation and small error
/// for every encoding and similarity, tighter with more bits.
#[test]
fn estimates_track_exact_scores() {
    let dim = 64;
    let mut data = vectors(200, dim, 7);
    let queries = vectors(5, dim, 99);
    for sim in SIMS {
        if sim == Similarity::DotProduct {
            data = data.iter().map(|v| normalize(v)).collect();
        }
        let queries: Vec<Vec<f32>> = if sim == Similarity::DotProduct {
            queries.iter().map(|v| normalize(v)).collect()
        } else {
            queries.clone()
        };
        let mut centroid = vec![0f32; dim];
        for v in &data {
            for (c, x) in centroid.iter_mut().zip(v) {
                *c += x / data.len() as f32;
            }
        }
        for enc in ENCODINGS {
            let q = OsqFactory::new(enc).train(dim, sim, &[]).unwrap();
            assert_eq!(q.name(), NAME);
            assert_eq!(q.dimension(), dim);
            assert_eq!(q.similarity(), sim);
            assert!(q.uses_centroid());
            let mut codes = vec![0u8; q.code_len() * data.len()];
            for (v, out) in data.iter().zip(codes.chunks_exact_mut(q.code_len())) {
                q.encode(v, &centroid, out);
            }
            let mut block = Vec::new();
            q.write_block(&codes, &mut block);
            let codes = block;
            let mut err = 0f64;
            for query in &queries {
                let mut s = q.query_scorer(query);
                s.set_centroid(&centroid);
                let mut est = vec![0f32; data.len()];
                s.score_block(&codes, &mut est);
                for (v, e) in data.iter().zip(&est) {
                    let exact = sim.score(query, v);
                    err += ((exact - e) / exact.max(1e-3)).abs() as f64;
                }
            }
            let mean = err / (queries.len() * data.len()) as f64;
            let bound = if enc.bits() <= 2 { 0.25 } else { 0.05 };
            assert!(mean < bound, "{sim:?} {enc:?}: mean relative error {mean}");
        }
    }
}

#[test]
fn state_round_trips_and_rejects_garbage() {
    let f = OsqFactory::bbq();
    assert_eq!(f.encoding(), ScalarEncoding::SingleBitQueryNibble);
    for enc in ENCODINGS {
        let q = OsqFactory::new(enc)
            .train(16, Similarity::Euclidean, &[0.0; 32])
            .unwrap();
        let back = f.read(16, Similarity::Euclidean, &q.state()).unwrap();
        assert_eq!(back.code_len(), q.code_len());
        assert_eq!(back.state(), q.state());
    }
    assert!(matches!(
        f.read(16, Similarity::Euclidean, &[]),
        Err(Error::CorruptState(_))
    ));
    assert!(matches!(
        f.read(16, Similarity::Euclidean, &[9]),
        Err(Error::CorruptState(_))
    ));
    assert!(f.read(0, Similarity::Euclidean, &[3]).is_err());
    assert!(f.train(3, Similarity::Euclidean, &[0.0; 4]).is_err());
}

#[test]
fn bbq_code_is_one_bit_per_dimension_plus_corrections() {
    let q = OsqQuantizer::new(
        100,
        Similarity::Euclidean,
        ScalarEncoding::SingleBitQueryNibble,
    );
    assert_eq!(q.encoding(), ScalarEncoding::SingleBitQueryNibble);
    // 100 dims pad to 128 bits (16 bytes); 16 bytes of corrections.
    assert_eq!(q.code_len(), 16 + 16);
}

#[test]
fn euclidean_self_score_is_near_one_and_mip_scales() {
    let dim = 32;
    let v = vectors(1, dim, 3).remove(0);
    let centroid = vec![0.1f32; dim];
    for (sim, check) in [
        (Similarity::Euclidean, 0.9f32),
        (Similarity::MaximumInnerProduct, 1.0),
    ] {
        let q = OsqQuantizer::new(dim, sim, ScalarEncoding::UnsignedByte);
        let mut code = vec![0u8; q.code_len()];
        q.encode(&v, &centroid, &mut code);
        let mut s = q.query_scorer(&v);
        s.set_centroid(&centroid);
        assert!(s.score(&code) >= check, "{sim:?} {}", s.score(&code));
    }
}

#[test]
#[should_panic(expected = "code length")]
fn encode_checks_lengths() {
    let q = OsqQuantizer::new(8, Similarity::Euclidean, ScalarEncoding::UnsignedByte);
    q.encode(&[0.0; 8], &[0.0; 8], &mut [0u8; 3]);
}

#[test]
#[should_panic(expected = "centroid dimension")]
fn set_centroid_checks_length() {
    let q = OsqQuantizer::new(8, Similarity::Euclidean, ScalarEncoding::UnsignedByte);
    q.query_scorer(&[0.0; 8]).set_centroid(&[0.0; 4]);
}

/// The fast-scan block layout scores every code exactly as scoring it
/// alone does, for partial and whole blocks.
#[test]
fn fast_scan_blocks_score_like_single_codes() {
    for encoding in [
        ScalarEncoding::SingleBitQueryNibble,
        ScalarEncoding::DibitQueryNibble,
        ScalarEncoding::PackedNibble,
    ] {
        for sim in SIMS {
            for dim in [5, 100, 300] {
                let q = OsqQuantizer::new(dim, sim, encoding);
                let centroid = vectors(1, dim, 99).remove(0);
                for n in [0, 1, 31, 32, 33, 70] {
                    let docs = vectors(n, dim, 7 + n as u64);
                    let mut codes = vec![0u8; n * q.code_len()];
                    for (d, c) in docs.iter().zip(codes.chunks_exact_mut(q.code_len())) {
                        q.encode(d, &centroid, c);
                    }
                    let mut block = Vec::new();
                    q.write_block(&codes, &mut block);
                    assert_eq!(block.len(), q.block_len(n));
                    let query = vectors(1, dim, 1234).remove(0);
                    let mut scorer = q.query_scorer(&query);
                    scorer.set_centroid(&centroid);
                    let mut scores = vec![0f32; n];
                    scorer.score_block(&block, &mut scores);
                    for (i, c) in codes.chunks_exact(q.code_len()).enumerate() {
                        assert_eq!(
                            scores[i].to_bits(),
                            scorer.score(c).to_bits(),
                            "{encoding:?} {sim:?} dim {dim} n {n} code {i}"
                        );
                    }
                }
            }
        }
    }
}
