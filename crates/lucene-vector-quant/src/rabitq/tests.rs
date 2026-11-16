#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::osq::OsqFactory;
use crate::QuantizerFactory;
use lucene_util::quantization::ScalarEncoding;

const SIMS: [Similarity; 4] = [
    Similarity::Euclidean,
    Similarity::DotProduct,
    Similarity::Cosine,
    Similarity::MaximumInnerProduct,
];

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f32 {
        (next(&mut self.0) >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gaussian(&mut self) -> f32 {
        let (u, v) = (self.uniform().max(1e-7), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// Gaussian vectors around a common offset; when `skewed`, with
/// per-dimension scales decaying geometrically and every 17th dimension
/// ten times larger.
fn gaussians(n: usize, dim: usize, seed: u64, skewed: bool) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|d| {
                    let scale = if !skewed {
                        1.0
                    } else if d % 17 == 3 {
                        10.0 * 0.97f32.powi(d as i32)
                    } else {
                        0.97f32.powi(d as i32)
                    };
                    0.5 + scale * rng.gaussian()
                })
                .collect()
        })
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn mean(data: &[Vec<f32>]) -> Vec<f32> {
    let mut c = vec![0f32; data[0].len()];
    for v in data {
        for (c, x) in c.iter_mut().zip(v) {
            *c += x / data.len() as f32;
        }
    }
    c
}

/// The raw similarity (squared distance or inner product) a score maps
/// back to.
fn raw(sim: Similarity, score: f32) -> f64 {
    let s = f64::from(score);
    match sim {
        Similarity::Euclidean => 1.0 / s - 1.0,
        Similarity::DotProduct | Similarity::Cosine => 2.0 * s - 1.0,
        Similarity::MaximumInnerProduct => {
            if s < 1.0 {
                1.0 - 1.0 / s
            } else {
                s - 1.0
            }
        }
    }
}

/// Mean absolute error of the raw estimates, relative to the mean
/// absolute raw value, of `factory` over `data` and `queries`.
fn error(
    factory: &dyn QuantizerFactory,
    sim: Similarity,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> f64 {
    let dim = data[0].len();
    let q = factory.train(dim, sim, &[]).unwrap();
    let centroid = mean(data);
    let mut codes = vec![0u8; q.code_len() * data.len()];
    for (v, out) in data.iter().zip(codes.chunks_exact_mut(q.code_len())) {
        q.encode(v, &centroid, out);
    }
    let mut block = Vec::new();
    q.write_block(&codes, &mut block);
    let (mut err, mut scale) = (0f64, 0f64);
    for query in queries {
        let mut s = q.query_scorer(query);
        s.set_centroid(&centroid);
        let mut est = vec![0f32; data.len()];
        s.score_block(&block, &mut est);
        for (v, &e) in data.iter().zip(&est) {
            let exact = raw(sim, sim.score(query, v));
            err += (raw(sim, e) - exact).abs();
            scale += exact.abs();
        }
    }
    err / scale
}

fn dataset(sim: Similarity, n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    skewed_or_not(sim, n, dim, seed, true)
}

fn skewed_or_not(sim: Similarity, n: usize, dim: usize, seed: u64, skewed: bool) -> Vec<Vec<f32>> {
    let data = gaussians(n, dim, seed, skewed);
    if matches!(sim, Similarity::DotProduct | Similarity::Cosine) {
        data.iter().map(|v| normalize(v)).collect()
    } else {
        data
    }
}

/// Error falls with every added bit, for every similarity, and is at or
/// near OSQ's at the bit widths both have on isotropic data, far below it
/// on skewed data.
#[test]
fn error_falls_with_bits_and_beats_osq() {
    let dim = 128;
    for (sim, skewed) in SIMS.into_iter().flat_map(|s| [(s, true), (s, false)]) {
        let data = skewed_or_not(sim, 300, dim, 7, skewed);
        let queries = skewed_or_not(sim, 8, dim, 99, skewed);
        let errs: Vec<f64> = (2..=8)
            .map(|b| error(&ExRabitqFactory::new(b), sim, &data, &queries))
            .collect();
        for w in errs.windows(2) {
            assert!(w[1] < w[0], "{sim:?} skewed {skewed}: {errs:?}");
        }
        for (bits, enc) in [
            (2, ScalarEncoding::DibitQueryNibble),
            (4, ScalarEncoding::PackedNibble),
            (7, ScalarEncoding::SevenBit),
            (8, ScalarEncoding::UnsignedByte),
        ] {
            let osq = error(&OsqFactory::new(enc), sim, &data, &queries);
            let erq = errs[bits - 2];
            println!(
                "{sim:?} skewed {skewed} {bits} bits: erq {erq:.5} osq {osq:.5} all {errs:.5?}"
            );
            // On isotropic data 2 bits tie and 8 bits (8-bit queries
            // both) come within 4%; elsewhere erq is well below.
            assert!(erq < osq * 1.05, "{sim:?} {bits} bits: erq {erq} osq {osq}");
        }
    }
}

/// Averaged over rotations, the inner-product estimate's error is
/// centred on zero: its mean is a small fraction of its spread.
#[test]
fn estimator_is_unbiased() {
    let dim = 64;
    let mut rng = Rng(5);
    let centroid = vec![0f32; dim];
    let doc: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
    let query: Vec<f32> = (0..dim).map(|_| rng.gaussian()).collect();
    let exact = f64::from(simd::dot_f32(&doc, &query));
    for bits in [2, 3] {
        let (mut sum, mut sq) = (0f64, 0f64);
        let trials = 2000;
        for seed in 0..trials {
            let q =
                ExRabitqQuantizer::new(dim, Similarity::MaximumInnerProduct, bits, seed).unwrap();
            let mut code = vec![0u8; q.code_len()];
            q.encode(&doc, &centroid, &mut code);
            let mut s = q.query_scorer(&query);
            s.set_centroid(&centroid);
            let e = raw(Similarity::MaximumInnerProduct, s.score(&code)) - exact;
            sum += e;
            sq += e * e;
        }
        let mean = sum / trials as f64;
        let std = (sq / trials as f64 - mean * mean).sqrt();
        let stderr = std / (trials as f64).sqrt();
        assert!(std > 0.0);
        assert!(
            mean.abs() < 4.0 * stderr,
            "{bits} bits: bias {mean} stderr {stderr}"
        );
    }
}

/// Every similarity and odd dimension scores close to the exact score,
/// including a vector against itself.
#[test]
fn all_similarities_and_dimensions() {
    for dim in [1, 7, 64, 100, 300] {
        for sim in SIMS {
            let data = dataset(sim, 40, dim, 3 + dim as u64);
            let queries = dataset(sim, 3, dim, 11);
            let e = error(&ExRabitqFactory::new(6), sim, &data, &queries);
            assert!(e < 0.02, "{sim:?} dim {dim}: {e}");
            let q = ExRabitqFactory::new(4).train(dim, sim, &[]).unwrap();
            assert_eq!((q.name(), q.dimension(), q.similarity()), (NAME, dim, sim));
            assert!(q.uses_centroid());
            let padded = code_dimension(dim);
            assert_eq!(
                padded,
                if dim <= 32 {
                    dim.next_power_of_two().max(8)
                } else {
                    dim.next_multiple_of(32)
                }
            );
            assert_eq!(q.code_len(), padded / 2 + CORRECTIONS_BYTES);
        }
    }
}

/// One dimension is rotated over eight code dimensions and estimated
/// to within a percent at 2 bits.
#[test]
fn dimension_one_estimates_closely() {
    let q = ExRabitqQuantizer::new(1, Similarity::Euclidean, 2, 1).unwrap();
    let mut code = vec![0u8; q.code_len()];
    q.encode(&[3.0], &[1.0], &mut code);
    let mut s = q.query_scorer(&[-1.0]);
    s.set_centroid(&[1.0]);
    let d2 = raw(Similarity::Euclidean, s.score(&code));
    assert!((d2 - 16.0).abs() < 0.16, "{d2}");
}

#[test]
fn state_round_trips_and_rejects_garbage() {
    let f = ExRabitqFactory::with_seed(5, 42);
    assert_eq!(f.bits(), 5);
    let q = f.train(10, Similarity::Cosine, &[0.0; 20]).unwrap();
    assert_eq!(q.state().len(), 9);
    let back = ExRabitqFactory::new(3)
        .read(10, Similarity::Cosine, &q.state())
        .unwrap();
    assert_eq!(back.state(), q.state());
    assert_eq!(back.code_len(), q.code_len());
    let v: Vec<f32> = (0..10).map(|i| i as f32 - 4.0).collect();
    let c = vec![0.1f32; 10];
    let (mut a, mut b) = (vec![0u8; q.code_len()], vec![0u8; q.code_len()]);
    q.encode(&v, &c, &mut a);
    back.encode(&v, &c, &mut b);
    assert_eq!(a, b);
    let typed = ExRabitqQuantizer::new(10, Similarity::Cosine, 5, 42).unwrap();
    assert_eq!((typed.bits(), typed.seed()), (5, 42));
    for bad in [
        &[][..],
        &[4, 0, 0][..],
        &[1, 0, 0, 0, 0, 0, 0, 0, 0][..],
        &[9; 9][..],
    ] {
        assert!(matches!(
            f.read(10, Similarity::Cosine, bad),
            Err(Error::CorruptState(_))
        ));
    }
    assert!(f.read(0, Similarity::Cosine, &q.state()).is_err());
    assert!(matches!(
        f.read(MAX_PADDED_DIMENSION + 1, Similarity::Cosine, &q.state()),
        Err(Error::IllegalArgument(_))
    ));
    assert!(f.train(3, Similarity::Cosine, &[0.0; 4]).is_err());
    assert!(f
        .train(MAX_PADDED_DIMENSION + 1, Similarity::Cosine, &[])
        .is_err());
    assert!(ExRabitqQuantizer::new(8, Similarity::Cosine, 1, 0).is_err());
    assert!(ExRabitqQuantizer::new(0, Similarity::Cosine, 4, 0).is_err());
    assert!(ExRabitqQuantizer::new(usize::MAX, Similarity::Cosine, 4, 0).is_err());
}

#[test]
#[should_panic(expected = "bits must be in 2..=8")]
fn factory_rejects_bits() {
    ExRabitqFactory::new(9);
}

/// The rotation keeps norms and inner products.
#[test]
fn rotation_is_orthogonal() {
    let q = ExRabitqQuantizer::new(100, Similarity::Euclidean, 4, 3).unwrap();
    let mut rng = Rng(8);
    let a: Vec<f32> = (0..100).map(|_| rng.gaussian()).collect();
    let b: Vec<f32> = (0..100).map(|_| rng.gaussian()).collect();
    let (mut ra, mut rb) = (vec![0f32; 128], vec![0f32; 128]);
    q.rotate(&a, &mut ra);
    q.rotate(&b, &mut rb);
    let (d, rd) = (simd::dot_f32(&a, &b), simd::dot_f32(&ra, &rb));
    assert!((d - rd).abs() < 1e-3, "{d} {rd}");
    let (n, rn) = (simd::dot_f32(&a, &a), simd::dot_f32(&ra, &ra));
    assert!((n - rn).abs() < 1e-3 * n);
}

/// The windowed level search finds the codes the full search does, and
/// their cosine to the vector grows with the bits as the paper reports
/// (about 0.94 at 2 bits, 0.98 at 3, 0.995 at 4).
#[test]
fn search_window_loses_nothing() {
    let mut rng = Rng(21);
    let mut search = Search::default();
    let min_cos = [0.0, 0.0, 0.92, 0.97, 0.99, 0.997, 0.999, 0.9997, 0.9999];
    for bits in 2..=8u8 {
        let max_level = (1u8 << (bits - 1)) - 1;
        for n in [64, 128, 512, 1024] {
            for _ in 0..8 {
                let a: Vec<f32> = (0..n).map(|_| rng.gaussian().abs()).collect();
                let cos = |levels: &[u8]| {
                    let y: Vec<f32> = levels.iter().map(|&l| l as f32 + 0.5).collect();
                    simd::dot_f32(&y, &a) / (simd::dot_f32(&y, &y) * simd::dot_f32(&a, &a)).sqrt()
                };
                let (mut full, mut fast) = (vec![0u8; n], vec![0u8; n]);
                best_levels(&a, max_level, FULL_WINDOW, &mut full, &mut search);
                let t = best_levels(
                    &a,
                    max_level,
                    SEARCH_WINDOW[usize::from(bits)],
                    &mut fast,
                    &mut search,
                );
                assert_eq!(full, fast, "{bits} bits n {n} t {t}");
                assert!(full.iter().all(|&l| l <= max_level));
                assert!(cos(&full) > min_cos[usize::from(bits)], "{bits} bits n {n}");
            }
        }
    }
    let mut zero = [9u8; 4];
    assert_eq!(
        best_levels(&[0.0; 4], 3, FULL_WINDOW, &mut zero, &mut search),
        0.0
    );
    assert_eq!(zero, [0; 4]);
}

#[test]
fn zero_residual_scores_the_centroid() {
    let q = ExRabitqQuantizer::new(16, Similarity::Euclidean, 3, 0).unwrap();
    let c = vec![0.5f32; 16];
    let mut code = vec![0u8; q.code_len()];
    q.encode(&c, &c, &mut code);
    let mut s = q.query_scorer(&c);
    s.set_centroid(&c);
    assert_eq!(s.score(&code), 1.0);
}

/// The fast-scan block layout scores every code exactly as scoring it
/// alone does, for partial and whole blocks.
#[test]
fn blocks_score_like_single_codes() {
    for bits in 2..=8 {
        for sim in SIMS {
            for dim in [5, 100] {
                let q = ExRabitqQuantizer::new(dim, sim, bits, 3).unwrap();
                let centroid = gaussians(1, dim, 99, false).remove(0);
                for n in [0, 1, 31, 32, 33, 70] {
                    let docs = gaussians(n, dim, 7 + n as u64, true);
                    let mut codes = vec![0u8; n * q.code_len()];
                    for (d, c) in docs.iter().zip(codes.chunks_exact_mut(q.code_len())) {
                        q.encode(d, &centroid, c);
                    }
                    let mut block = Vec::new();
                    q.write_block(&codes, &mut block);
                    assert_eq!(block.len(), q.block_len(n));
                    let query = gaussians(1, dim, 1234, true).remove(0);
                    let mut scorer = q.query_scorer(&query);
                    scorer.set_centroid(&centroid);
                    let mut scores = vec![0f32; n];
                    scorer.score_block(&block, &mut scores);
                    for (i, c) in codes.chunks_exact(q.code_len()).enumerate() {
                        assert_eq!(
                            scores[i].to_bits(),
                            scorer.score(c).to_bits(),
                            "{bits} bits {sim:?} dim {dim} n {n} code {i}"
                        );
                    }
                }
            }
        }
    }
}

/// Prepared centroids encode and score as plain ones do.
#[test]
fn prepared_centroids_match_plain_ones() {
    let q = ExRabitqQuantizer::new(20, Similarity::DotProduct, 3, 1).unwrap();
    let c = gaussians(1, 20, 5, false).remove(0);
    let v = gaussians(1, 20, 6, false).remove(0);
    let prepared = q.prepare_centroid(&c).unwrap();
    assert_eq!(prepared.len(), 32);
    let (mut a, mut b) = (vec![0u8; q.code_len()], vec![0u8; q.code_len()]);
    q.encode(&v, &c, &mut a);
    q.encode_prepared(&v, &prepared, &mut b);
    assert_eq!(a, b);
    let mut s = q.query_scorer(&v);
    s.set_centroid(&c);
    let plain = s.score(&a);
    s.set_prepared_centroid(&prepared);
    assert_eq!(plain, s.score(&a));
}

/// A zero residual takes the lowest non-negative level everywhere,
/// `u = 2^(B-1)`: in bit planes only the top plane is set; in bytes every
/// byte holds it.
#[test]
fn layouts_hold_codes() {
    for bits in [3, 8] {
        let q = ExRabitqQuantizer::new(8, Similarity::Euclidean, bits, 0).unwrap();
        let mut code = vec![0u8; q.code_len()];
        q.encode(&[1.0; 8], &[1.0; 8], &mut code);
        let half = 1u8 << (bits - 1);
        if bits <= MAX_PLANE_BITS {
            for (p, &plane) in code[..usize::from(bits)].iter().enumerate() {
                let want = if p == usize::from(bits) - 1 { 0xff } else { 0 };
                assert_eq!(plane, want, "plane {p}");
            }
        } else {
            assert_eq!(&code[..8], &[half; 8]);
        }
    }
}

/// Times scoring and encoding against OSQ at 128 dimensions. Run with
/// `cargo test --release -p lucene-vector-quant -- --ignored --nocapture microbench`.
#[test]
#[ignore]
fn microbench() {
    use std::time::Instant;
    let dim = 128;
    let n = 256;
    let data = gaussians(n, dim, 1, true);
    let centroid = mean(&data);
    let query = gaussians(1, dim, 2, true).remove(0);
    let mut factories: Vec<(String, Box<dyn QuantizerFactory>)> = vec![
        (
            "osq2".into(),
            Box::new(OsqFactory::new(ScalarEncoding::DibitQueryNibble)),
        ),
        (
            "osq4".into(),
            Box::new(OsqFactory::new(ScalarEncoding::PackedNibble)),
        ),
        (
            "osq7".into(),
            Box::new(OsqFactory::new(ScalarEncoding::SevenBit)),
        ),
    ];
    for b in 2..=8 {
        factories.push((format!("erq{b}"), Box::new(ExRabitqFactory::new(b))));
    }
    for (name, f) in &factories {
        let q = f.train(dim, Similarity::Euclidean, &[]).unwrap();
        let prepared = q
            .prepare_centroid(&centroid)
            .unwrap_or_else(|| centroid.clone());
        let mut codes = vec![0u8; n * q.code_len()];
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            for (v, c) in data.iter().zip(codes.chunks_exact_mut(q.code_len())) {
                q.encode_prepared(v, &prepared, c);
            }
        }
        let encode = t.elapsed().as_secs_f64() * 1e6 / (reps * n) as f64;
        let mut block = Vec::new();
        q.write_block(&codes, &mut block);
        let mut s = q.query_scorer(&query);
        let mut scores = vec![0f32; n];
        let reps = 4000;
        let t = Instant::now();
        for _ in 0..reps {
            s.set_prepared_centroid(&prepared);
            s.score_block(&block, &mut scores);
            std::hint::black_box(&scores);
        }
        let probe = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            s.set_prepared_centroid(&prepared);
            std::hint::black_box(&s);
        }
        let set = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!(
                "{name}: encode {encode:.2} us/vector, probe of {n} codes {probe:.2} us ({:.1} ns/code, {set:.2} us to set the centroid)",
                probe * 1e3 / n as f64
            );
    }
}

#[test]
#[should_panic(expected = "code length")]
fn encode_checks_lengths() {
    let q = ExRabitqQuantizer::new(8, Similarity::Euclidean, 4, 0).unwrap();
    q.encode(&[0.0; 8], &[0.0; 8], &mut [0u8; 3]);
}

#[test]
#[should_panic(expected = "centroid dimension")]
fn set_centroid_checks_length() {
    let q = ExRabitqQuantizer::new(8, Similarity::Euclidean, 4, 0).unwrap();
    q.query_scorer(&[0.0; 8]).set_centroid(&[0.0; 4]);
}
