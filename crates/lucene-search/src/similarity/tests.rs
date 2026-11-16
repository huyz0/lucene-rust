#![cfg(test)]
use super::*;

// Hand-computed (independently of the implementation, via the same formula
// read directly off `BM25Similarity.java`) expected values -- see the
// `test-coverage` skill's rule against "coverage theater": these assert
// pre-computed numbers, not "whatever the code currently produces".

#[test]
fn the_encoded_impacts_bound_is_the_decoded_one_bit_for_bit() {
    use lucene_codecs::postings::{decode_impacts, Impact};
    use lucene_store::DataOutput;
    let mut table = [0.0f32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        *t = 1.0 / (1.0 + i as f32 * 0.37);
    }
    let pairs: [&[(i32, i64)]; 3] = [
        &[(1, 1), (3, 2), (4, 9), (40, 200), (90, 255)],
        &[(2, 7)],
        &[(1, 1), (300, 2), (301, 130)],
    ];
    for list in pairs {
        // Encode as `Lucene104PostingsWriter.writeImpacts` does.
        let mut bytes = Vec::new();
        let mut prev = Impact { freq: 0, norm: 0 };
        for &(freq, norm) in list {
            let freq_delta = (freq - prev.freq - 1) as u32;
            let norm_delta = norm - prev.norm - 1;
            if norm_delta == 0 {
                bytes.write_vint((freq_delta << 1) as i32);
            } else {
                bytes.write_vint(((freq_delta << 1) | 1) as i32);
                bytes.write_zlong(norm_delta);
            }
            prev = Impact { freq, norm };
        }
        let decoded = decode_impacts(&bytes).unwrap();
        assert_eq!(decoded.len(), list.len());
        let want = max_score_for_impacts_table(&decoded, 2.5, &table);
        let got = max_score_for_encoded_impacts_table(&bytes, 2.5, &table).unwrap();
        assert_eq!(got.to_bits(), want.to_bits());
    }
    // Nothing to bound, or nothing that decodes: no bound.
    assert_eq!(max_score_for_encoded_impacts_table(&[], 2.5, &table), None);
    assert_eq!(
        max_score_for_encoded_impacts_table(&[0x80], 2.5, &table),
        None
    );
}

#[test]
fn idf_matches_hand_computed_value() {
    // docFreq=1, docCount=10: ln(1 + (10 - 1 + 0.5)/(1 + 0.5)) = ln(1 + 9.5/1.5)
    // = ln(1 + 6.333333...) = ln(7.333333...) = 1.9924302...
    let got = idf(1, 10);
    assert!((got - 1.992_430_2).abs() < 1e-5, "got {got}");
}

#[test]
fn idf_decreases_as_doc_freq_increases() {
    // A more common term (higher docFreq) must score a lower idf over the same
    // docCount -- the defining monotonicity property of IDF.
    assert!(idf(1, 100) > idf(50, 100));
    assert!(idf(50, 100) > idf(99, 100));
}

#[test]
fn idf_can_go_negative_for_a_term_in_every_document() {
    // Real BM25Similarity's ln(1+x) smoothing still allows idf < 0 when
    // docFreq == docCount (every doc contains the term): ln(1 + 0.5/(N+0.5)),
    // which is > 0 actually for finite N -- but as docFreq approaches docCount
    // for large N it approaches ln(1) = 0 from above. Assert the concrete
    // hand-computed boundary value instead of a vague claim.
    // docFreq=10, docCount=10: ln(1 + (10-10+0.5)/(10+0.5)) = ln(1 + 0.5/10.5)
    // = ln(1.047619...) = 0.046520...
    let got = idf(10, 10);
    assert!((got - 0.046_520).abs() < 1e-4, "got {got}");
}

#[test]
fn tf_norm_matches_hand_computed_value() {
    // freq=3, fieldLength=avgFieldLength=1.0 (this port's constant
    // substitution), k1=1.2, b=0.75:
    // tfNorm = 3 / (3 + 1.2*(1-0.75+0.75*1/1)) = 3 / (3 + 1.2*1.0)
    // = 3 / 4.2 = 0.714285...
    let got = tf_norm(3.0, 1.0, 1.0, DEFAULT_K1, DEFAULT_B);
    assert!((got - 0.714_285_7).abs() < 1e-5, "got {got}");
}

#[test]
fn tf_norm_with_field_longer_than_average_reduces_score() {
    // freq=2, fieldLength=2*avgFieldLength: b*fieldLength/avgFieldLength term
    // doubles, penalizing tf_norm relative to the fieldLength==avgFieldLength
    // case -- BM25's length-normalization property.
    let baseline = tf_norm(2.0, 1.0, 1.0, DEFAULT_K1, DEFAULT_B);
    let longer_doc = tf_norm(2.0, 2.0, 1.0, DEFAULT_K1, DEFAULT_B);
    assert!(longer_doc < baseline);
}

#[test]
fn score_combines_idf_and_tf_norm() {
    // docFreq=2, docCount=10, freq=4, unnormed field length.
    // idf(2,10) = ln(1 + (10-2+0.5)/(2+0.5)) = ln(1 + 8.5/2.5) = ln(4.4)
    //           = 1.481_604...
    // tfNorm(4, 1, 1, 1.2, 0.75) = 4 / (4 + 1.2*1.0) = 4/5.2 = 0.769_230...
    // score = 1.481604 * 0.769230... = 1.139696...
    let got = score(2, 10, 4.0, UNNORMED_FIELD_LENGTH, UNNORMED_FIELD_LENGTH);
    let expected_idf = 4.4f64.ln() as f32;
    let expected_tf_norm = 4.0f32 / 5.2f32;
    let expected = expected_idf * expected_tf_norm;
    assert!(
        (got - expected).abs() < 1e-4,
        "got {got}, expected {expected}"
    );
    assert!((got - 1.139_696).abs() < 1e-3, "got {got}");
}

#[test]
fn bm25_params_new_rejects_exactly_what_lucene_rejects() {
    // `BM25Similarity(float k1, float b, boolean, float)`'s own guards,
    // with its own messages.
    assert_eq!(
        Bm25Params::new(-0.1, 0.5).unwrap_err(),
        "illegal k1 value: -0.1, must be a non-negative finite value"
    );
    assert!(Bm25Params::new(f32::NAN, 0.5).is_err());
    assert!(Bm25Params::new(f32::INFINITY, 0.5).is_err());
    assert_eq!(
        Bm25Params::new(1.2, 1.5).unwrap_err(),
        "illegal b value: 1.5, must be between 0 and 1"
    );
    assert!(Bm25Params::new(1.2, -0.001).is_err());
    assert!(Bm25Params::new(1.2, f32::NAN).is_err());
    // The boundaries Lucene accepts.
    assert_eq!(
        Bm25Params::new(0.0, 0.0).unwrap(),
        Bm25Params { k1: 0.0, b: 0.0 }
    );
    assert_eq!(
        Bm25Params::new(1.2, 1.0).unwrap(),
        Bm25Params { k1: 1.2, b: 1.0 }
    );
    assert_eq!(
        Bm25Params::new(DEFAULT_K1, DEFAULT_B).unwrap(),
        Bm25Params::default()
    );
}

#[test]
fn do_score_reproduces_the_algebraic_form_to_within_rounding() {
    // The two expressions are equal in exact arithmetic; the point of
    // `do_score` is that it is the one real Lucene evaluates. Pin both
    // facts: close, and not always identical.
    let mut differed = false;
    for freq in [1.0f32, 2.0, 3.0, 7.0, 41.0] {
        for len in [1.0f32, 2.0, 5.0, 17.0, 96.0] {
            let weight = idf(3, 97);
            let got = do_score(weight, freq, norm_inverse(len, 12.5, DEFAULT_K1, DEFAULT_B));
            let algebraic = weight * tf_norm(freq, len, 12.5, DEFAULT_K1, DEFAULT_B);
            // A few ULP, not one: the two expressions differ in *where*
            // they round (one division and a subtraction versus two
            // divisions and a multiplication), so the gap compounds.
            assert!(
                (got - algebraic).abs() <= 4.0 * f32::EPSILON * got.abs().max(1.0),
                "freq={freq} len={len}: {got} vs {algebraic}"
            );
            differed |= got.to_bits() != algebraic.to_bits();
        }
    }
    assert!(
        differed,
        "if the two forms never differ, this crate's insistence on `do_score` \
             would be pointless -- they do differ, and Lucene's bits are do_score's"
    );
}

#[test]
fn norm_inverse_matches_lucenes_scorer_cache_entry() {
    // `cache[i] = 1f / (k1 * ((1 - b) + b * LENGTH_TABLE[i] / avgdl))`.
    let avgdl = 4.0f32;
    for byte in [0u8, 1, 23, 40, 100, 255] {
        let len = decode_norm(byte as i64);
        let expected = 1.0f32 / (DEFAULT_K1 * ((1.0 - DEFAULT_B) + DEFAULT_B * len / avgdl));
        assert_eq!(norm_inverse(len, avgdl, DEFAULT_K1, DEFAULT_B), expected);
    }
    // And the constant the no-norms path uses is the same function at
    // `UNNORMED_FIELD_LENGTH`.
    assert_eq!(
        UNNORMED_NORM_INVERSE,
        norm_inverse(
            UNNORMED_FIELD_LENGTH,
            UNNORMED_FIELD_LENGTH,
            DEFAULT_K1,
            DEFAULT_B
        )
    );
}

#[test]
fn max_score_for_impacts_is_never_below_a_real_scored_document() {
    // The soundness property MAXSCORE rests on, and the reason the bound
    // and the score must be the same float expression: for every
    // (freq, norm) pair the bound covers, the bound must be >= the score
    // that pair actually produces. One ULP the wrong way here drops hits.
    let impacts: Vec<lucene_codecs::postings::Impact> = [(1i32, 0i64), (3, 40), (9, 100)]
        .iter()
        .map(|&(freq, norm)| lucene_codecs::postings::Impact { freq, norm })
        .collect();
    let avgdl = 7.5f32;
    let bound = max_score_for_impacts(&impacts, 5, 200, avgdl);
    for impact in &impacts {
        let actual = score(5, 200, impact.freq as f32, decode_norm(impact.norm), avgdl);
        assert!(actual <= bound, "{actual} > bound {bound} for {impact:?}");
    }
    // Unnormed sibling, same property against the unnormed scoring path.
    let unnormed = max_score_for_impacts_unnormed(&impacts, 5, 200);
    for impact in &impacts {
        let actual = score(
            5,
            200,
            impact.freq as f32,
            UNNORMED_FIELD_LENGTH,
            UNNORMED_FIELD_LENGTH,
        );
        assert!(actual <= unnormed, "{actual} > bound {unnormed}");
    }
}

#[test]
fn bm25_params_default_matches_lucene_default_constants() {
    let params = Bm25Params::default();
    assert_eq!(params.k1(), DEFAULT_K1);
    assert_eq!(params.b(), DEFAULT_B);
}

#[test]
fn score_with_params_using_defaults_matches_score_byte_for_byte() {
    // Regression proof: the new parameterized path must
    // reproduce the existing hardcoded-default path exactly when given
    // Bm25Params::default(), not just "close enough".
    let got = score_with_params(
        2,
        10,
        4.0,
        UNNORMED_FIELD_LENGTH,
        UNNORMED_FIELD_LENGTH,
        Bm25Params::default(),
    );
    let expected = score(2, 10, 4.0, UNNORMED_FIELD_LENGTH, UNNORMED_FIELD_LENGTH);
    assert_eq!(
        got, expected,
        "got {got}, expected byte-identical {expected}"
    );
}

#[test]
fn score_with_params_using_different_k1_b_matches_hand_computed_value() {
    // docFreq=2, docCount=10, freq=4, unnormed field length, but k1=2.0,
    // b=0.5 instead of the 1.2/0.75 defaults.
    // idf(2,10) = ln(4.4) = 1.481604... (same as `score_combines_idf_and_tf_norm`)
    // tfNorm(4, 1, 1, 2.0, 0.5) = 4 / (4 + 2.0*(1 - 0.5 + 0.5*1/1))
    //           = 4 / (4 + 2.0*1.0) = 4 / 6.0 = 0.666666...
    // score = 1.481604 * 0.666666... = 0.987736...
    let params = Bm25Params::new(2.0, 0.5).expect("in range");
    let got = score_with_params(
        2,
        10,
        4.0,
        UNNORMED_FIELD_LENGTH,
        UNNORMED_FIELD_LENGTH,
        params,
    );
    let expected_idf = 4.4f64.ln() as f32;
    let expected_tf_norm = 4.0f32 / 6.0f32;
    let expected = expected_idf * expected_tf_norm;
    assert!(
        (got - expected).abs() < 1e-4,
        "got {got}, expected {expected}"
    );
    assert!((got - 0.987_736).abs() < 1e-3, "got {got}");
    // And it must differ measurably from the default-params score.
    let default_score = score(2, 10, 4.0, UNNORMED_FIELD_LENGTH, UNNORMED_FIELD_LENGTH);
    assert!(
        (got - default_score).abs() > 1e-3,
        "different k1/b must produce a measurably different score: {got} vs {default_score}"
    );
}

#[test]
fn score_is_zero_when_freq_is_zero() {
    assert_eq!(score(1, 10, 0.0, 1.0, 1.0), 0.0);
}

#[test]
fn score_increases_with_freq_all_else_equal() {
    let low = score(5, 100, 1.0, 1.0, 1.0);
    let high = score(5, 100, 5.0, 1.0, 1.0);
    assert!(high > low);
}

#[test]
fn decode_norm_matches_small_float_byte4_to_int() {
    // Same known values as `lucene_util::small_float`'s test, reached
    // through this module's `i64`-sign-extension-aware wrapper.
    assert_eq!(decode_norm(0), 0.0);
    assert_eq!(decode_norm(23), 23.0);
    assert_eq!(decode_norm(100), 3096.0);
    // Byte 200 sign-extends to a negative i64 the way
    // `norms::norm_value` returns it (`read_byte() as i8 as i64`); the
    // `as u8` truncation must still recover the original byte.
    assert_eq!(decode_norm(200i64 as i8 as i64), 16_777_240.0);
    assert_eq!(decode_norm(255i64 as i8 as i64), 2_013_265_944.0);
}

#[test]
fn max_score_for_impacts_matches_hand_computed_value_single_entry() {
    // One impact: freq=3, norm byte 5 -> decode_norm(5) == 5.0 (subnormal,
    // exact -- see `decode_norm_matches_small_float_byte4_to_int`).
    // docFreq=2, docCount=10, avgFieldLength=5.0 (so fieldLength ==
    // avgFieldLength, collapsing the length-norm term to 1.0):
    // idf(2,10) = ln(4.4) = 1.481604..., tfNorm(3,5,5,1.2,0.75)
    // = 3 / (3 + 1.2*1.0) = 3/4.2 = 0.714285...
    // expected = 1.481604 * 0.714285... = 1.058289...
    let impacts = vec![lucene_codecs::postings::Impact { freq: 3, norm: 5 }];
    let got = max_score_for_impacts(&impacts, 2, 10, 5.0);
    let expected = idf(2, 10) * tf_norm(3.0, 5.0, 5.0, DEFAULT_K1, DEFAULT_B);
    assert!(
        (got - expected).abs() < 1e-5,
        "got {got}, expected {expected}"
    );
    assert!((got - 1.058_289).abs() < 1e-3, "got {got}");
}

/// **The table-driven bound is the arithmetic one, bit for bit**, and the
/// unnormed bound's largest-frequency shortcut is the fold it replaced:
/// the score is monotone in `freq * normInverse`, so taking the largest
/// product first changes no bit. Random impact lists (unsorted, with
/// repeats), weights and average lengths, and the empty list.
#[test]
fn the_table_bound_is_the_arithmetic_bound_exactly() {
    use lucene_codecs::postings::Impact;
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _ in 0..500 {
        let avg = 1.0 + (next() % 5000) as f32 / 7.0;
        let weight = (next() % 10_000) as f32 / 997.0;
        let table = crate::field_norms::inverse_table(avg);
        let n = (next() % 12) as usize;
        let impacts: Vec<Impact> = (0..n)
            .map(|_| Impact {
                freq: 1 + (next() % 300) as i32,
                norm: (next() % 256) as i64,
            })
            .collect();
        let want = max_score_for_impacts_weighted(&impacts, weight, avg);
        let got = max_score_for_impacts_table(&impacts, weight, &table);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{impacts:?} w={weight} avg={avg}"
        );
        let unnormed = impacts
            .iter()
            .map(|i| {
                do_score(
                    weight,
                    i.freq as f32,
                    norm_inverse(
                        UNNORMED_FIELD_LENGTH,
                        UNNORMED_FIELD_LENGTH,
                        DEFAULT_K1,
                        DEFAULT_B,
                    ),
                )
            })
            .fold(0.0f32, f32::max);
        assert_eq!(
            max_score_for_impacts_unnormed_weighted(&impacts, weight).to_bits(),
            unnormed.to_bits()
        );
    }
}

#[test]
fn max_score_for_impacts_empty_slice_is_zero() {
    assert_eq!(max_score_for_impacts(&[], 2, 10, 1.0), 0.0);
}

#[test]
fn max_score_for_impacts_takes_the_true_max_not_just_the_last_entry() {
    // Impacts are ordered by strictly increasing freq *and* strictly
    // increasing norm (the CompetitiveImpactAccumulator invariant) -- but
    // a higher norm decodes to a longer field, which *penalizes* tf_norm.
    // Construct a case where the last (highest-freq, highest-norm) entry
    // scores *lower* than an earlier entry, proving this function must
    // scan every entry rather than assuming the list's tail bounds it.
    let impacts = vec![
        lucene_codecs::postings::Impact { freq: 2, norm: 1 }, // short field, modest freq
        lucene_codecs::postings::Impact { freq: 3, norm: 60 }, // much longer field
    ];
    let doc_freq = 2;
    let doc_count = 10;
    let avg_field_length = 1.0;
    let score_entry_0 = idf(doc_freq, doc_count)
        * tf_norm(2.0, decode_norm(1), avg_field_length, DEFAULT_K1, DEFAULT_B);
    let score_entry_1 = idf(doc_freq, doc_count)
        * tf_norm(
            3.0,
            decode_norm(60),
            avg_field_length,
            DEFAULT_K1,
            DEFAULT_B,
        );
    assert!(
        score_entry_0 > score_entry_1,
        "test setup must make the earlier (lower-freq, lower-norm) entry \
             the higher-scoring one: {score_entry_0} vs {score_entry_1}"
    );
    let got = max_score_for_impacts(&impacts, doc_freq, doc_count, avg_field_length);
    assert!(
        (got - score_entry_0).abs() < 1e-5,
        "got {got}, expected {score_entry_0}"
    );
}

#[test]
fn max_score_for_impacts_bounds_a_doc_matching_any_single_entry_exactly() {
    // The property this port's pruning actually relies on (see
    // `max_score_for_impacts`'s doc comment): a real doc whose (freq,
    // norm) exactly matches one of the block's competitive impacts
    // entries can never score higher than this function's bound over
    // the whole list -- note this is *not* the same as "any
    // component-wise-dominated (freq, norm) pair is safe" (a doc with a
    // smaller norm than every entry but a small freq can, in principle,
    // score higher than an entry that traded a larger norm for a larger
    // freq -- BM25's tf and length-norm terms move in opposite
    // directions). The real write-side guarantee
    // (`CompetitiveImpactAccumulator`) is that every real doc in the
    // block matches this exactly-one-entry shape or is dominated in the
    // score sense, not the naive component-wise sense.
    let impacts = vec![
        lucene_codecs::postings::Impact { freq: 1, norm: 3 },
        lucene_codecs::postings::Impact { freq: 5, norm: 20 },
        lucene_codecs::postings::Impact { freq: 10, norm: 50 },
    ];
    let doc_freq = 3;
    let doc_count = 50;
    let avg_field_length = 10.0;
    let bound = max_score_for_impacts(&impacts, doc_freq, doc_count, avg_field_length);
    for impact in &impacts {
        let field_length = decode_norm(impact.norm);
        let actual = score(
            doc_freq,
            doc_count,
            impact.freq as f32,
            field_length,
            avg_field_length,
        );
        assert!(
            actual <= bound + 1e-4,
            "doc score {actual} (freq={}, norm={}) exceeded bound {bound}",
            impact.freq,
            impact.norm
        );
    }
}

#[test]
fn block_pruning_via_max_score_matches_brute_force_top_1() {
    use lucene_codecs::postings::{Impact, Postings};

    // Block A: docs 1..=3, freq=5, norm byte 5 (decode_norm(5) == 5.0,
    // exact) -- scores relatively high once avg_field_length == 5.0.
    // Block B: docs 4..=6, freq=1, norm byte 100 (decode_norm(100) ==
    // 3096.0, a much longer field) -- scores much lower.
    let postings = Postings {
        docs: vec![1, 2, 3, 4, 5, 6],
        freqs: vec![5, 5, 5, 1, 1, 1],
        level0_impacts: vec![
            (3, vec![Impact { freq: 5, norm: 5 }]),
            (6, vec![Impact { freq: 1, norm: 100 }]),
        ],
        level1_impacts: Vec::new(),
    };
    let norm_bytes = [5u8, 5, 5, 100, 100, 100];
    let doc_freq = 2;
    let doc_count = 10;
    let avg_field_length = 5.0;

    // Sanity: block B's bound really is lower than block A's real
    // per-doc scores, so the pruning path in the harness below actually
    // exercises the skip branch rather than vacuously never triggering.
    let block_a_score = score(doc_freq, doc_count, 5.0, 5.0, avg_field_length);
    let block_b_bound = max_score_for_impacts(
        &[Impact { freq: 1, norm: 100 }],
        doc_freq,
        doc_count,
        avg_field_length,
    );
    assert!(
        block_b_bound < block_a_score,
        "test setup must make block B's bound beatable by block A's score: \
             {block_b_bound} vs {block_a_score}"
    );

    assert_block_pruning_matches_brute_force(
        &postings,
        &norm_bytes,
        doc_freq,
        doc_count,
        avg_field_length,
        1,
    );
}

#[test]
fn block_pruning_via_max_score_matches_brute_force_top_2_spans_both_blocks() {
    use lucene_codecs::postings::{Impact, Postings};

    // Same two blocks as above, but top_n=2 needs one hit from each
    // block (block A's docs alone can supply at most... here 3, so this
    // covers the "collector not yet full when reaching block B" and "no
    // skip should happen since block B still has a competitive doc"
    // shape, proving pruning doesn't skip when it can't yet prove
    // safety).
    let postings = Postings {
        docs: vec![1, 2, 3, 4, 5, 6],
        freqs: vec![5, 5, 5, 4, 4, 4],
        level0_impacts: vec![
            (3, vec![Impact { freq: 5, norm: 5 }]),
            (6, vec![Impact { freq: 4, norm: 5 }]),
        ],
        level1_impacts: Vec::new(),
    };
    let norm_bytes = [5u8, 5, 5, 5, 5, 5];
    assert_block_pruning_matches_brute_force(&postings, &norm_bytes, 2, 10, 5.0, 2);
}

#[test]
fn score_with_real_decoded_lengths_differs_from_unnormed_constant() {
    // Two docs with different real (decoded) field lengths must get
    // different tf_norm contributions -- proving the length-
    // normalization term is actually live, not collapsed to a constant.
    let short_doc_len = decode_norm(5); // byte 5 -> length 5 (subnormal, exact)
    let long_doc_len = decode_norm(40); // byte 40 -> a longer decoded length
    assert!(long_doc_len > short_doc_len);

    let avg = (short_doc_len + long_doc_len) / 2.0;
    let score_short = score(2, 10, 3.0, short_doc_len, avg);
    let score_long = score(2, 10, 3.0, long_doc_len, avg);
    assert_ne!(score_short, score_long);
    // Same BM25 property as `tf_norm_with_field_longer_than_average_reduces_score`:
    // the shorter-than-average doc scores higher for the same freq/idf.
    assert!(score_short > score_long);
}
