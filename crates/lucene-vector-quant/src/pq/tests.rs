#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::{QuantizerFactory, QuantizerRegistry, Similarity};

/// `n` deterministic vectors of `dim` floats, clustered so codebooks have
/// structure to learn.
fn sample(n: usize, dim: usize) -> Vec<f32> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
            out.push(((i % 8) as f32 - 3.5) * if j % 2 == 0 { 1.0 } else { -0.5 } + noise * 0.2);
        }
    }
    out
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn trained(sim: Similarity) -> Box<dyn Quantizer> {
    PqFactory::new(4).train(8, sim, &sample(1_000, 8)).unwrap()
}

#[test]
fn shape_of_a_trained_quantizer() {
    let q = trained(Similarity::Euclidean);
    assert_eq!((q.name(), q.dimension(), q.code_len()), ("pq", 8, 4));
    assert!(q.uses_centroid());
    assert_eq!(q.similarity(), Similarity::Euclidean);
    // m, then m * 256 * dsub floats.
    assert_eq!(q.state().len(), 4 + 4 * 256 * 2 * 4);
}

#[test]
fn training_is_deterministic() {
    let a = trained(Similarity::Euclidean).state();
    assert_eq!(a, trained(Similarity::Euclidean).state());
}

#[test]
fn state_round_trips() {
    let q = trained(Similarity::Euclidean);
    let back = PqFactory::new(1)
        .read(8, Similarity::Euclidean, &q.state())
        .unwrap();
    assert_eq!(back.state(), q.state());
    let v = &sample(3, 8)[8..16];
    let (mut a, mut b) = (vec![0u8; 4], vec![0u8; 4]);
    q.encode(v, &[0.0; 8], &mut a);
    back.encode(v, &[0.0; 8], &mut b);
    assert_eq!(a, b);
}

#[test]
fn reconstruction_is_close_and_adc_matches_the_decoded_distance() {
    let q = trained(Similarity::Euclidean);
    let data = sample(64, 8);
    let centroid = [0.25f32; 8];
    let query = &data[8..16];
    let mut scorer = q.query_scorer(query);
    scorer.set_centroid(&centroid);
    let pq = q.as_ref();
    let mut total_err = 0.0;
    for v in data.chunks_exact(8) {
        let mut code = vec![0u8; 4];
        pq.encode(v, &centroid, &mut code);
        let decoded = decode(&q.state(), 8, &code, &centroid);
        // The books were trained on the vectors themselves, so the residual
        // against the zero centroid is what they reconstruct.
        let mut zero_code = vec![0u8; 4];
        pq.encode(v, &[0.0; 8], &mut zero_code);
        total_err += dist2(v, &decode(&q.state(), 8, &zero_code, &[0.0; 8]));
        let want = 1.0 / (1.0 + dist2(query, &decoded));
        assert!(
            (scorer.score(&code) - want).abs() < 1e-4,
            "{} vs {want}",
            scorer.score(&code)
        );
    }
    // Residuals of eight tight blobs, 256 entries a subspace: near exact.
    assert!(total_err / 64.0 < 0.05, "mean error {}", total_err / 64.0);
}

#[test]
fn inner_product_scores_follow_the_decoded_dot() {
    for sim in [
        Similarity::DotProduct,
        Similarity::MaximumInnerProduct,
        Similarity::Cosine,
    ] {
        let q = trained(sim);
        let data = sample(16, 8);
        let (query, centroid) = (&data[8..16], [0.1f32; 8]);
        let mut scorer = q.query_scorer(query);
        scorer.set_centroid(&centroid);
        for v in data.chunks_exact(8) {
            let mut code = vec![0u8; 4];
            q.encode(v, &centroid, &mut code);
            let (mut qn, mut vn) = (query.to_vec(), v.to_vec());
            if sim == Similarity::Cosine {
                lucene_util::vector_util::l2normalize(&mut qn, false).unwrap();
                lucene_util::vector_util::l2normalize(&mut vn, false).unwrap();
            }
            let decoded = decode(
                &q.state(),
                8,
                &code_of(q.as_ref(), &vn, &centroid, sim),
                &centroid,
            );
            let raw = dot(&qn, &decoded);
            let want = match sim {
                Similarity::MaximumInnerProduct => {
                    lucene_util::quantization::scale_max_inner_product_score(raw)
                }
                _ => ((1.0 + raw) / 2.0).max(0.0),
            };
            assert!((scorer.score(&code) - want).abs() < 1e-3, "{sim:?}");
        }
    }
}

/// The code of an already-normalised vector (cosine encodes normalise
/// again, which is idempotent).
fn code_of(q: &dyn Quantizer, v: &[f32], c: &[f32], _s: Similarity) -> Vec<u8> {
    let mut code = vec![0u8; q.code_len()];
    q.encode(v, c, &mut code);
    code
}

/// Codebook entries a code names, plus the centroid.
fn decode(state: &[u8], dim: usize, code: &[u8], centroid: &[f32]) -> Vec<f32> {
    let m = u32::from_le_bytes(state[..4].try_into().unwrap()) as usize;
    let dsub = dim / m;
    let book = |s: usize, j: usize, i: usize| {
        let at = 4 + ((s * 256 + j) * dsub + i) * 4;
        f32::from_le_bytes(state[at..at + 4].try_into().unwrap())
    };
    (0..dim)
        .map(|k| centroid[k] + book(k / dsub, code[k / dsub] as usize, k % dsub))
        .collect()
}

#[test]
fn block_scoring_matches_single_scoring() {
    let q = trained(Similarity::Euclidean);
    let data = sample(10, 8);
    let c = [0.0f32; 8];
    let mut codes = Vec::new();
    for v in data.chunks_exact(8) {
        let mut code = vec![0u8; 4];
        q.encode(v, &c, &mut code);
        codes.extend(code);
    }
    let mut scorer = q.query_scorer(&data[..8]);
    scorer.set_centroid(&c);
    let mut block = vec![0f32; 10];
    scorer.score_block(&codes, &mut block);
    for (i, s) in block.iter().enumerate() {
        assert_eq!(s.to_bits(), scorer.score(&codes[i * 4..][..4]).to_bits());
    }
}

#[test]
fn a_new_centroid_retargets_the_table() {
    let q = trained(Similarity::Euclidean);
    let data = sample(4, 8);
    let (c1, c2) = ([0.0f32; 8], [1.0f32; 8]);
    let mut code = vec![0u8; 4];
    q.encode(&data[..8], &c1, &mut code);
    let mut s = q.query_scorer(&data[8..16]);
    s.set_centroid(&c1);
    let first = s.score(&code);
    s.set_centroid(&c2);
    assert_ne!(first, s.score(&code));
    s.set_centroid(&c1);
    assert_eq!(first.to_bits(), s.score(&code).to_bits());
    s.set_prepared_centroid(&c1);
    assert_eq!(first.to_bits(), s.score(&code).to_bits());
}

#[test]
fn training_refuses_unusable_input() {
    let f = PqFactory::new(4);
    let ok = sample(300, 8);
    let msg = |m: usize, d: usize, s: &[f32]| {
        PqFactory::new(m)
            .train(d, Similarity::Euclidean, s)
            .map(|_| ())
            .unwrap_err()
            .to_string()
    };
    assert!(msg(0, 8, &ok).contains("m must be positive"));
    assert!(msg(3, 8, &ok).contains("divide"));
    assert!(msg(4, 0, &ok).contains("dimension"));
    assert!(msg(4, 8, &ok[..9]).contains("whole"));
    assert!(msg(4, 8, &ok[..8 * 255]).contains(
        "Number of training points (255) should be at least as large as number of clusters (256)"
    ));
    let mut bad = ok.clone();
    bad[5] = f32::NAN;
    assert!(msg(4, 8, &bad).contains("finite"));
    assert_eq!(f.name(), "pq");
    assert_eq!(f.code_dimension(8), 8);
}

#[test]
fn reading_refuses_garbage() {
    let q = trained(Similarity::Euclidean);
    let state = q.state();
    let f = PqFactory::new(1);
    let read = |d: usize, s: &[u8]| f.read(d, Similarity::Euclidean, s).map(|_| ()).unwrap_err();
    assert!(matches!(read(8, &[]), Error::CorruptState(_)));
    assert!(matches!(
        read(8, &state[..state.len() - 1]),
        Error::CorruptState(_)
    ));
    assert!(
        matches!(read(6, &state), Error::CorruptState(_)),
        "m must divide dimension"
    );
    let mut zero = state.clone();
    zero[..4].copy_from_slice(&0u32.to_le_bytes());
    assert!(matches!(read(8, &zero), Error::CorruptState(_)));
    let mut nan = state.clone();
    nan[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(matches!(read(8, &nan), Error::CorruptState(_)));
    assert!(matches!(read(0, &state), Error::IllegalArgument(_)));
}

#[test]
#[should_panic(expected = "vector dimension")]
fn encode_checks_the_vector() {
    trained(Similarity::Euclidean).encode(&[0.0; 7], &[0.0; 8], &mut [0u8; 4]);
}

#[test]
#[should_panic(expected = "code length")]
fn encode_checks_the_output() {
    trained(Similarity::Euclidean).encode(&[0.0; 8], &[0.0; 8], &mut [0u8; 3]);
}

#[test]
#[should_panic(expected = "centroid dimension")]
fn encode_checks_the_centroid() {
    trained(Similarity::Euclidean).encode(&[0.0; 8], &[0.0; 7], &mut [0u8; 4]);
}

#[test]
#[should_panic(expected = "query dimension")]
fn scorer_checks_the_query() {
    let _ = trained(Similarity::Euclidean).query_scorer(&[0.0; 7]);
}

#[test]
fn kmeans_finds_the_blobs_and_splits_empty_clusters() {
    // Two blobs and k = 4: two centroids start empty and must be split off.
    let mut x = Vec::new();
    for i in 0..40 {
        x.extend_from_slice(&[(i % 5) as f32 * 0.01, 0.0]);
        x.extend_from_slice(&[10.0 + (i % 5) as f32 * 0.01, 10.0]);
    }
    let c = kmeans::train(&x, 2, 4, 9);
    assert_eq!(c.len(), 8);
    assert!(c.iter().all(|v| v.is_finite()));
    let near = |p: [f32; 2]| {
        c.chunks_exact(2)
            .map(|cc| dist2(cc, &p))
            .fold(f32::INFINITY, f32::min)
    };
    assert!(near([0.02, 0.0]) < 0.01);
    assert!(near([10.02, 10.0]) < 0.01);
}

#[test]
fn registry_reads_pq_back() {
    let r = QuantizerRegistry::with_builtins();
    let q = trained(Similarity::Euclidean);
    let back = r
        .get("pq")
        .unwrap()
        .read(8, Similarity::Euclidean, &q.state())
        .unwrap();
    assert_eq!(back.state(), q.state());
}
