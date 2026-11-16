#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;

#[test]
fn registry_finds_replaces_and_rejects() {
    let mut r = QuantizerRegistry::with_builtins();
    assert_eq!(r.names(), vec!["osq", "sq", "erq", "rot-osq", "rot-sq"]);
    assert!(r.get("osq").is_ok());
    assert!(r.get("rot-sq").is_ok());
    assert_eq!(
        r.get("rabitq").unwrap_err(),
        Error::UnknownQuantizer("rabitq".into())
    );
    r.register(Arc::new(osq::OsqFactory::bbq()));
    assert_eq!(r.names(), vec!["sq", "erq", "rot-osq", "rot-sq", "osq"]);
    assert!(QuantizerRegistry::new().names().is_empty());
}

#[test]
fn centroids_are_prepared_as_they_are_by_default() {
    let q = osq::OsqFactory::bbq()
        .train(4, Similarity::Euclidean, &[])
        .unwrap();
    let (v, c) = ([1.0, -2.0, 0.5, 3.0], [0.25, 0.0, -1.0, 1.0]);
    assert_eq!(q.prepare_centroid(&c), None);
    let (mut a, mut b) = (vec![0u8; q.code_len()], vec![0u8; q.code_len()]);
    q.encode(&v, &c, &mut a);
    q.encode_prepared(&v, &c, &mut b);
    assert_eq!(a, b);
    let mut s = q.query_scorer(&v);
    s.set_centroid(&c);
    let plain = s.score(&a);
    s.set_centroid(&[0.0; 4]);
    s.set_prepared_centroid(&c);
    assert_eq!(s.score(&a).to_bits(), plain.to_bits());
}

#[test]
fn sample_checks() {
    assert!(check_sample(0, &[]).is_err());
    assert!(check_sample(3, &[1.0; 4]).is_err());
    assert!(check_sample(2, &[1.0; 4]).is_ok());
}

#[test]
fn prepare_normalizes_cosine_only() {
    let mut out = [0f32; 2];
    prepare_vector(Similarity::Cosine, &[3.0, 4.0], &mut out);
    assert!((out[0] - 0.6).abs() < 1e-6 && (out[1] - 0.8).abs() < 1e-6);
    prepare_vector(Similarity::DotProduct, &[3.0, 4.0], &mut out);
    assert_eq!(out, [3.0, 4.0]);
    prepare_vector(Similarity::Cosine, &[0.0, 0.0], &mut out);
    assert_eq!(out, [0.0, 0.0]);
}

#[test]
fn score_margins() {
    use Similarity::*;
    // Euclidean: score 0.5 is d^2 = 1; 0.5 closer is score 1 / 1.5.
    assert!((score_margin(Euclidean, 0.5, 0.5) - (1.0 / 1.5 - 0.5)).abs() < 1e-6);
    assert_eq!(score_margin(Euclidean, 0.5, f32::INFINITY), 0.5);
    assert_eq!(score_margin(Euclidean, 0.0, 1.0), 0.0);
    for sim in [DotProduct, Cosine] {
        assert!((score_margin(sim, 0.75, 0.2) - 0.1).abs() < 1e-6);
        // A clamped score is taken at the clamp, raw -1.
        assert!((score_margin(sim, 0.0, 0.5) - 0.25).abs() < 1e-6);
        assert_eq!(score_margin(sim, 0.5, f32::INFINITY), f32::INFINITY);
    }
    // MIP: raw 1 scores 2, raw 1.5 scores 2.5; raw -1 scores 0.5.
    assert!((score_margin(MaximumInnerProduct, 2.0, 0.5) - 0.5).abs() < 1e-6);
    assert!((score_margin(MaximumInnerProduct, 0.5, 2.0) - 1.5).abs() < 1e-6);
    assert_eq!(score_margin(DotProduct, 0.5, f32::NAN), f32::INFINITY);
    assert_eq!(score_margin(DotProduct, 0.5, -1.0), 0.0);
    assert_eq!(score_margin(DotProduct, f32::NAN, 0.1), f32::INFINITY);
}

fn gaussian_vectors(n: usize, dim: usize, sim: Similarity, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim)
            .map(|d| {
                let g = next() + next() + next() + next();
                0.3 + g * if d % 5 == 0 { 3.0 } else { 1.0 }
            })
            .collect();
        if matches!(sim, Similarity::DotProduct | Similarity::Cosine) {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
        }
        out.extend(v);
    }
    out
}

/// The share of codes whose exact score exceeds estimate plus margin,
/// and the mean margin, of `factory` over clustered data; `None` when
/// the scorer has no bounds.
fn bound_misses(factory: &dyn QuantizerFactory, sim: Similarity, dim: usize) -> Option<(f64, f64)> {
    let n = 600;
    let data = gaussian_vectors(n, dim, sim, 5);
    let queries = gaussian_vectors(12, dim, sim, 6);
    let q = factory.train(dim, sim, &data).unwrap();
    let mut centroid = vec![0f32; dim];
    for v in data.chunks_exact(dim) {
        for (c, x) in centroid.iter_mut().zip(v) {
            *c += x / n as f32;
        }
    }
    let prepared = q.prepare_centroid(&centroid);
    let target = prepared.as_deref().unwrap_or(&centroid);
    let mut codes = vec![0u8; q.code_len() * n];
    for (v, out) in data
        .chunks_exact(dim)
        .zip(codes.chunks_exact_mut(q.code_len()))
    {
        q.encode_prepared(v, target, out);
    }
    let mut block = Vec::new();
    q.write_block(&codes, &mut block);
    let (mut misses, mut total, mut margin) = (0usize, 0usize, 0f64);
    for query in queries.chunks_exact(dim) {
        let mut s = q.query_scorer(query);
        s.set_prepared_centroid(target);
        let mut est = vec![0f32; n];
        s.score_block(&block, &mut est);
        let mut bounds = vec![0f32; n];
        if !s.error_bounds(&block, &est, &mut bounds) {
            assert_eq!(s.error_bound(&block, 0, est[0]), None);
            return None;
        }
        // One code's margin is the block's.
        for j in [0, 31, 32, n - 1] {
            assert_eq!(s.error_bound(&block, j, est[j]), Some(bounds[j]));
        }
        for ((v, &e), &b) in data.chunks_exact(dim).zip(&est).zip(&bounds) {
            assert!(b >= 0.0 && b.is_finite(), "{} {sim:?}: bound {b}", q.name());
            let exact = sim.score(query, v);
            misses += usize::from(exact > e + b + 1e-6);
            total += 1;
            margin += f64::from(b);
        }
    }
    Some((misses as f64 / total as f64, margin / total as f64))
}

#[test]
fn error_bounds_cover_the_exact_scores() {
    use lucene_util::quantization::ScalarEncoding as E;
    let factories: Vec<(Box<dyn QuantizerFactory>, f64)> = vec![
        (Box::new(osq::OsqFactory::bbq()), 0.03),
        (Box::new(osq::OsqFactory::new(E::DibitQueryNibble)), 0.03),
        (Box::new(osq::OsqFactory::new(E::PackedNibble)), 0.03),
        (Box::new(osq::OsqFactory::new(E::SevenBit)), 0.03),
        (Box::new(osq::OsqFactory::new(E::UnsignedByte)), 0.03),
        (Box::new(scalar::ScalarFactory::int7()), 0.01),
        (Box::new(scalar::ScalarFactory::int4()), 0.01),
        (Box::new(rabitq::ExRabitqFactory::new(2)), 0.03),
        (Box::new(rabitq::ExRabitqFactory::new(4)), 0.03),
        (Box::new(rabitq::ExRabitqFactory::new(7)), 0.03),
        (
            Box::new(rotation::RotatedFactory::new(Arc::new(
                osq::OsqFactory::bbq(),
            ))),
            0.03,
        ),
    ];
    for sim in [
        Similarity::Euclidean,
        Similarity::DotProduct,
        Similarity::Cosine,
        Similarity::MaximumInnerProduct,
    ] {
        for (f, allowed) in &factories {
            for dim in [37, 128] {
                let got = bound_misses(f.as_ref(), sim, dim);
                match got {
                    Some((share, _)) => assert!(
                        share <= *allowed,
                        "{} {sim:?} {dim}: {share} of exact scores above the bound",
                        f.name()
                    ),
                    None => {
                        assert!(f.name() == rabitq::NAME && sim == Similarity::MaximumInnerProduct)
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct Constant;
impl QueryScorer for Constant {
    fn set_centroid(&mut self, _: &[f32]) {}
    fn score(&self, code: &[u8]) -> f32 {
        code.len() as f32
    }
}

#[test]
fn default_score_block() {
    let mut scores = [0f32; 3];
    Constant.score_block(&[0; 6], &mut scores);
    assert_eq!(scores, [2.0; 3]);
    Constant.score_block(&[], &mut scores);
    assert_eq!(scores, [0.0; 3]);
    let mut none: [f32; 0] = [];
    Constant.score_block(&[], &mut none);
}

#[test]
fn bounds_are_unknown_by_default() {
    assert!(!Constant.error_bounds(&[0; 4], &[1.0; 2], &mut [0.0; 2]));
    assert_eq!(Constant.error_bound(&[0; 4], 1, 1.0), None);
}

#[test]
#[should_panic(expected = "one code per score")]
fn score_block_rejects_ragged_codes() {
    Constant.score_block(&[0; 5], &mut [0f32; 2]);
}
