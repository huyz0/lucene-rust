#![cfg(test)]
use super::*;

fn norm(num_terms: u32) -> i64 {
    i64::from(small_float::int_to_byte4(num_terms) as i8)
}

fn stats(term: &[u8], df: i64, ttf: i64) -> TermStatistics {
    TermStatistics::new(df, ttf).unwrap().with_term(term)
}

fn field_stats() -> CollectionStatistics {
    CollectionStatistics::new(4, 4, 3003, 2000).unwrap()
}

fn freq(v: i64) -> Explanation {
    Explanation::match_long(v, "freq")
}

#[test]
fn illegal_k1() {
    for k1 in [f32::INFINITY, -1.0, f32::NAN] {
        let e = StableTflSimilarity::new(k1, 0.917).unwrap_err();
        assert!(e.contains("illegal k1 value"), "{e}");
    }
}

#[test]
fn illegal_c() {
    for c in [f32::INFINITY, -1.0, f32::NAN] {
        let e = StableTflSimilarity::new(1.2, c).unwrap_err();
        assert!(e.contains("illegal c value"), "{e}");
    }
    assert_eq!(
        StableTflSimilarity::new(1.2, -1.0).unwrap_err(),
        "illegal c value: -1.0, must be a non-negative finite value"
    );
}

#[test]
fn valid_parameters() {
    let sim = StableTflSimilarity::new(0.0, 0.0).unwrap();
    assert_eq!((sim.k1(), sim.c()), (0.0, 0.0));
    let sim = StableTflSimilarity::new(1.5, 0.9).unwrap();
    assert_eq!((sim.k1(), sim.c()), (1.5, 0.9));
    let defaults = StableTflSimilarity::default();
    assert_eq!(defaults.k1(), StableTflSimilarity::DEFAULT_K1);
    assert_eq!(defaults.c(), StableTflSimilarity::DEFAULT_C);
    assert_eq!(defaults.simple_name(), "StableTflSimilarity");
    assert!(defaults.shared().is_some());
}

#[test]
fn to_string() {
    assert_eq!(
        StableTflSimilarity::default().to_string(),
        "StableTflSimilarity(k1=1.2, c=0.917)"
    );
}

/// `testExplain`: the canonical example, to the float bit and the
/// explanation string.
#[test]
fn explain() {
    let sim = StableTflSimilarity::default();
    let scorer = sim.scorer(
        "field",
        1.0,
        &field_stats(),
        &[stats(b"photosynthesis", 3, 3)],
    );
    let norm = norm(1000);
    let score = scorer.score(1.0, norm);
    assert_eq!(score.to_bits(), 1.395_581_1f32.to_bits());
    let explain = scorer.explain(freq(1), norm);
    assert_eq!(explain.value.to_bits(), score.to_bits());
    let expected = "\
1.3955811 = score(freq=1), computed as boost * tr * tf from:
  1.0 = boost
  0.45454544 = tf, computed as freq / (freq + k1) from:
    1 = freq
    1.2 = k1, term saturation parameter
  3.0702786 = tr, term rarity, computed as log(1 + (1 - p + 0.05) / (p + 0.05)) from:
    0.001049047818598714 = p, probability that the term appears in the doc, \
computed as 1 - (1 - m * 2 ^ (-c * tl)) ^ dl from:
      0.00781 = m, multiplicative constant to term match probability
      0.917 = c, decaying constant for term length
      14.0 = tl, term length
      984.0 = dl, document length
";
    assert_eq!(explain.to_string(), expected);
}

/// `testMultiTermExplain`.
#[test]
fn multi_term_explain() {
    let sim = StableTflSimilarity::default();
    let scorer = sim.scorer(
        "field",
        1.0,
        &field_stats(),
        &[stats(b"photosynthesis", 3, 3), stats(b"chlorophyll", 3, 3)],
    );
    let norm = norm(1000);
    let explain = scorer.explain(freq(1), norm);
    assert!(explain
        .to_string()
        .contains("tr, term rarity, computed as the sum of:"));
    assert_eq!(explain.value.to_bits(), scorer.score(1.0, norm).to_bits());
}

/// `testNonUtf8TermFallsBackToDefaultLength`, plus a truncated sequence
/// and a term-less statistic, which take the same default.
#[test]
fn non_utf8_term_falls_back_to_default_length() {
    let sim = StableTflSimilarity::default();
    let norm = norm(1000);
    let score = |t: TermStatistics| {
        sim.scorer("field", 1.0, &field_stats(), &[t])
            .score(1.0, norm)
    };
    let five = score(stats(b"hello", 3, 3));
    assert_eq!(score(stats(&[0xff, 0xff], 3, 3)).to_bits(), five.to_bits());
    assert_eq!(
        score(stats(&[b'a', 0xe2, 0x82], 3, 3)).to_bits(),
        five.to_bits()
    );
    assert_eq!(
        score(TermStatistics::new(3, 3).unwrap()).to_bits(),
        five.to_bits()
    );
    // Code points, not bytes: six bytes, five code points.
    assert_eq!(
        score(stats("hellö".as_bytes(), 3, 3)).to_bits(),
        five.to_bits()
    );
    assert_ne!(score(stats(b"hello!", 3, 3)).to_bits(), five.to_bits());
    assert_eq!(code_point_count("hellö€𝄞".as_bytes()), Some(7));
    assert_eq!(code_point_count(&[0x80]), None);
    assert_eq!(code_point_count(&[0xf8]), None);
}

/// `testScoreIndependentOfCorpusStats`.
#[test]
fn score_independent_of_corpus_stats() {
    let sim = StableTflSimilarity::default();
    let norm = norm(1000);
    let sparse = sim.scorer(
        "field",
        1.0,
        &field_stats(),
        &[stats(b"photosynthesis", 1, 1)],
    );
    let dense = sim.scorer(
        "field",
        1.0,
        &CollectionStatistics::new(1_000_000, 999_999, 123_456_789, 99_999_999).unwrap(),
        &[stats(b"photosynthesis", 999_999, 12_345_678)],
    );
    assert_eq!(
        sparse.score(1.0, norm).to_bits(),
        dense.score(1.0, norm).to_bits()
    );
}

/// `testMultiTermScoreIsSumOfSingleTermScores`.
#[test]
fn multi_term_score_is_sum_of_single_term_scores() {
    let sim = StableTflSimilarity::default();
    let (t1, t2) = (stats(b"photosynthesis", 3, 3), stats(b"chlorophyll", 3, 3));
    let norm = norm(1000);
    let freq = 2.0;
    let combined = sim
        .scorer("field", 1.0, &field_stats(), &[t1.clone(), t2.clone()])
        .score(freq, norm);
    let sum = sim
        .scorer("field", 1.0, &field_stats(), &[t1])
        .score(freq, norm)
        + sim
            .scorer("field", 1.0, &field_stats(), &[t2])
            .score(freq, norm);
    assert!((sum - combined).abs() <= 1e-6, "{sum} vs {combined}");
}

/// Java's class (`branch_10_6`, compiled against the 10.5.2 jars) over
/// a grid of parameters, terms (a multi-byte one summed with another),
/// boosts, norms and frequencies: the FNV-1a hash of every score's bits
/// it printed, and a two-term explanation.
#[test]
fn scores_and_explains_as_javas_class() {
    let params = [(1.2, 0.917), (0.5, 2.0), (3.0, 0.1), (0.0, 0.0)];
    let terms = ["a", "hello", "photosynthesis", "日本語"];
    let field = CollectionStatistics::new(4, 4, 3003, 2000).unwrap();
    let mut h = 0xcbf2_9ce4_8422_2325u64 as i64;
    let mut n = 0;
    for (k1, c) in params {
        let sim = StableTflSimilarity::new(k1, c).unwrap();
        for (t, term) in terms.iter().enumerate() {
            for boost in [1.0, 2.5] {
                let mut ts = vec![stats(term.as_bytes(), 1, 1)];
                if t == terms.len() - 1 {
                    ts.insert(0, stats(b"a", 1, 1));
                }
                let s = sim.scorer("field", boost, &field, &ts);
                for norm in (0..256).step_by(5) {
                    for f in [1.0, 3.0, 10.0, 1000.0] {
                        let bits = s.score(f, i64::from(norm as u8 as i8)).to_bits();
                        h = (h ^ i64::from(bits as i32)).wrapping_mul(0x100_0000_01b3);
                        n += 1;
                    }
                }
            }
        }
    }
    assert_eq!((n, h), (6656, -3_942_988_566_969_333_868));

    let s = StableTflSimilarity::default().scorer(
        "field",
        1.0,
        &field,
        &[stats(b"photosynthesis", 3, 3), stats(b"chlorophyll", 3, 3)],
    );
    let expected = "\
2.9462104 = score(freq=2), computed as boost * tr * tf from:
  1.0 = boost
  0.625 = tf, computed as freq / (freq + k1) from:
    2 = freq
    1.2 = k1, term saturation parameter
  4.7139365673065186 = tr, term rarity, computed as the sum of:
    2.7931688 = tr, term rarity, computed as log(1 + (1 - p + 0.05) / (p + 0.05)) from:
      0.017349573291813525 = p, probability that the term appears in the doc, \
computed as 1 - (1 - m * 2 ^ (-c * tl)) ^ dl from:
        0.00781 = m, multiplicative constant to term match probability
        0.917 = c, decaying constant for term length
        14.0 = tl, term length
        16408.0 = dl, document length
    1.9207678 = tr, term rarity, computed as log(1 + (1 - p + 0.05) / (p + 0.05)) from:
      0.11114389064304298 = p, probability that the term appears in the doc, \
computed as 1 - (1 - m * 2 ^ (-c * tl)) ^ dl from:
        0.00781 = m, multiplicative constant to term match probability
        0.917 = c, decaying constant for term length
        11.0 = tl, term length
        16408.0 = dl, document length
";
    assert_eq!(s.explain(freq(2), 120).to_string(), expected);
}

/// The boost is folded into the cache, and the bulk scorer agrees with
/// the scalar one.
#[test]
fn boost_and_bulk() {
    let sim = StableTflSimilarity::new(2.0, 0.5).unwrap();
    let one = sim.scorer("field", 1.0, &field_stats(), &[stats(b"abc", 1, 1)]);
    let two = sim.scorer("field", 3.0, &field_stats(), &[stats(b"abc", 1, 1)]);
    assert!(two.score(4.0, 7) > one.score(4.0, 7));
    let freqs = [1.0, 2.0, 30.0];
    let norms = [1, 50, -3];
    let mut scores = Vec::new();
    two.score_bulk(&freqs, &norms, &mut scores);
    for ((&f, &n), s) in freqs.iter().zip(&norms).zip(scores) {
        assert_eq!(s.to_bits(), two.score(f, n).to_bits());
    }
}
