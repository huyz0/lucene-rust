#![cfg(test)]
use super::*;

#[test]
fn float_similarities() {
    let a = [0.6f32, 0.8];
    let b = [1.0f32, 0.0];
    assert_eq!(
        VectorSimilarityFunction::Euclidean.score(&a, &b),
        1.0 / (1.0 + (0.16 + 0.64))
    );
    assert_eq!(VectorSimilarityFunction::DotProduct.score(&a, &b), 0.8);
    assert_eq!(VectorSimilarityFunction::Cosine.score(&a, &b), 0.8);
    assert_eq!(VectorSimilarityFunction::Cosine.score(&a, &[0.0, 0.0]), 0.5);
    assert_eq!(
        VectorSimilarityFunction::MaximumInnerProduct.score(&a, &b),
        1.6
    );
    assert_eq!(scale_max_inner_product_score(-1.0), 0.5);
}

#[test]
fn encodings_geometry_matches_java() {
    use ScalarEncoding::*;
    for e in ScalarEncoding::ALL {
        assert_eq!(ScalarEncoding::from_wire_number(e.wire_number()), Some(e));
    }
    assert_eq!(ScalarEncoding::from_wire_number(9), None);
    assert_eq!(ScalarEncoding::from_num_bits(4), Some(PackedNibble));
    assert_eq!(ScalarEncoding::from_num_bits(3), None);
    assert!(!UnsignedByte.is_asymmetric());
    assert!(SingleBitQueryNibble.is_asymmetric());
    // dims 13
    assert_eq!(UnsignedByte.discrete_dimensions(13), 13);
    assert_eq!(UnsignedByte.doc_packed_length(13), 13);
    assert_eq!(PackedNibble.discrete_dimensions(13), 14);
    assert_eq!(PackedNibble.doc_packed_length(13), 7);
    assert_eq!(SevenBit.doc_packed_length(13), 13);
    assert_eq!(SingleBitQueryNibble.discrete_dimensions(13), 16);
    assert_eq!(SingleBitQueryNibble.doc_packed_length(13), 2);
    assert_eq!(SingleBitQueryNibble.query_packed_length(13), 8);
    assert_eq!(DibitQueryNibble.discrete_dimensions(13), 16);
    assert_eq!(DibitQueryNibble.doc_packed_length(13), 4);
    assert_eq!(DibitQueryNibble.query_packed_length(13), 8);
    assert_eq!(UnsignedByte.query_packed_length(13), 13);
    assert_eq!(UnsignedByte.query_bits_per_dim(), 8);
}

#[test]
fn pack_unpack_round_trips() {
    let v: Vec<u8> = (0..24).map(|i| (i * 7 % 16) as u8).collect();
    let mut packed = vec![0u8; 12];
    pack_nibbles(&v, &mut packed);
    let mut back = vec![0u8; 24];
    unpack_nibbles(&packed, &mut back);
    assert_eq!(back, v);

    // Lucene 10.6's `TestVectorUtil.testInt4Unpack`: lengths around a
    // vector register and odd tails, against the scalar loop; bytes past
    // `2 * n` are left alone.
    for n in [0usize, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 2048] {
        let packed: Vec<u8> = (0..n).map(|i| (i * 151 + 29) as u8).collect();
        let mut actual = vec![0xAAu8; n * 2 + 1];
        unpack_nibbles(&packed, &mut actual);
        for (i, &p) in packed.iter().enumerate() {
            assert_eq!(actual[i], p >> 4, "n={n} i={i}");
            assert_eq!(actual[n + i], p & 0x0F, "n={n} i={i}");
        }
        assert_eq!(actual[n * 2], 0xAA);
    }

    let bits: Vec<u8> = (0..21).map(|i| (i % 3 == 0) as u8).collect();
    let mut packed = vec![0u8; 3];
    pack_as_binary(&bits, &mut packed);
    let mut back = vec![0u8; 21];
    unpack_binary(&packed, &mut back);
    assert_eq!(back, bits);

    let dibits: Vec<u8> = (0..21).map(|i| (i % 4) as u8).collect();
    let mut packed = vec![0u8; 6];
    transpose_dibit(&dibits, &mut packed);
    let mut back = vec![0u8; 21];
    untranspose_dibit(&packed, &mut back);
    assert_eq!(back, dibits);

    let q: Vec<u8> = (0..16).map(|i| (i % 16) as u8).collect();
    let mut t = vec![0u8; 8];
    transpose_half_byte(&q, &mut t);
    // plane 0 (low bits) of 0..16: odd values -> 0b01010101 twice.
    assert_eq!(&t[..2], &[0x55, 0x55]);
    assert_eq!(discretize(13, 8), 16);
    assert_eq!(discretize(16, 8), 16);
}

#[test]
fn quantize_dequantize_is_close() {
    let q = OptimizedScalarQuantizer::new(VectorSimilarityFunction::Euclidean);
    let orig: Vec<f32> = (0..32)
        .map(|i| ((i * 37 % 17) as f32 - 8.0) / 3.0)
        .collect();
    let centroid = vec![0.1f32; 32];
    for bits in [1u8, 2, 4, 7, 8] {
        let mut v = orig.clone();
        let mut dest = vec![0u8; 32];
        let r = q.scalar_quantize(&mut v, &mut dest, bits, &centroid);
        assert!(dest.iter().all(|&d| (d as u32) < (1 << bits)));
        assert_eq!(
            r.quantized_component_sum,
            dest.iter().map(|&d| d as i32).sum::<i32>()
        );
        let mut back = vec![0f32; 32];
        dequantize(
            &dest,
            &mut back,
            bits,
            r.lower_interval,
            r.upper_interval,
            &centroid,
        );
        if bits == 8 {
            for (a, b) in orig.iter().zip(&back) {
                assert!((a - b).abs() < 0.1, "{a} vs {b}");
            }
        }
    }
    let mut v = orig.clone();
    let mut dests = vec![vec![0u8; 32], vec![0u8; 32]];
    let rs = q.multi_scalar_quantize(&mut v, &mut dests, &[4, 8], &centroid);
    let mut v4 = orig.clone();
    let mut d4 = vec![0u8; 32];
    assert_eq!(rs[0], q.scalar_quantize(&mut v4, &mut d4, 4, &centroid));
    assert_eq!(dests[0], d4);
    // A constant vector (zero variance, zero norm) takes the early exit.
    let mut flat = vec![0.1f32; 8];
    let mut d = vec![0u8; 8];
    let r = q.scalar_quantize(&mut flat, &mut d, 4, &[0.1; 8]);
    assert_eq!(r.additional_correction, 0.0);
}

#[test]
fn legacy_quantizer_basics() {
    assert_eq!(
        ScalarQuantizer::new(f32::NAN, 1.0, 7),
        Err(QuantizerError::NonFinite)
    );
    let sq = ScalarQuantizer::new(-1.0, 1.0, 7).unwrap();
    assert_eq!(sq.bits(), 7);
    assert_eq!(sq.lower_quantile(), -1.0);
    assert_eq!(sq.upper_quantile(), 1.0);
    let alpha = 2.0f32 / 127.0;
    assert_eq!(sq.constant_multiplier(), alpha * alpha);
    let mut dest = [0u8; 3];
    let c = sq.quantize(
        &[-1.0, 0.0, 1.0],
        &mut dest,
        VectorSimilarityFunction::DotProduct,
    );
    assert_eq!(dest, [0, 64, 127]);
    assert!(c.is_finite());
    assert_eq!(
        sq.quantize(
            &[0.5, 0.0, 1.0],
            &mut dest,
            VectorSimilarityFunction::Euclidean
        ),
        0.0
    );
    let mut back = [0f32; 3];
    sq.dequantize(&dest, &mut back);
    assert!((back[2] - 1.0).abs() < 1e-6);
    let other = ScalarQuantizer::new(-2.0, 2.0, 7).unwrap();
    assert_eq!(
        other.recalculate_corrective_offset(&dest, &sq, VectorSimilarityFunction::Euclidean),
        0.0
    );
    assert!(other
        .recalculate_corrective_offset(&dest, &sq, VectorSimilarityFunction::DotProduct)
        .is_finite());
    assert_eq!(
        sq.to_string(),
        "ScalarQuantizer{minQuantile=-1, maxQuantile=1, bits=7}"
    );

    let vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| vec![i as f32 / 50.0, -(i as f32) / 25.0])
        .collect();
    let full = ScalarQuantizer::from_vectors(&vectors, 1.0, 50, 7).unwrap();
    assert_eq!(
        (full.lower_quantile(), full.upper_quantile()),
        (-49.0 / 25.0, 49.0 / 50.0)
    );
    let empty = ScalarQuantizer::from_vectors(&vectors[..0], 0.99, 0, 7).unwrap();
    assert_eq!(empty.lower_quantile(), 0.0);
    let ci = ScalarQuantizer::from_vectors(&vectors, 0.9, 50, 7).unwrap();
    assert!(ci.lower_quantile() > -49.0 / 25.0 && ci.upper_quantile() < 49.0 / 50.0);
    let auto = ScalarQuantizer::from_vectors_auto_interval(
        &vectors,
        VectorSimilarityFunction::DotProduct,
        50,
        4,
    )
    .unwrap();
    assert!(auto.lower_quantile() < auto.upper_quantile());
    assert_eq!(
        ScalarQuantizer::from_vectors_auto_interval(
            &vectors[..0],
            VectorSimilarityFunction::Euclidean,
            0,
            4
        )
        .unwrap()
        .upper_quantile(),
        0.0
    );
}

/// Past their sample size `fromVectors` and `fromVectorsAutoInterval`
/// read a reservoir sample: over copies of one vector every sample is
/// that vector, so they find what the same number of copies read whole
/// gives. Non-finite values leave the interval search no quantiles.
#[test]
fn sampled_quantiles_over_copies_of_one_vector_are_the_whole_sets() {
    let v: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) / 3.0).collect();
    let copies = |n: usize| vec![v.clone(); n];
    let (whole, many) = (copies(100), copies(1500));
    let read = |vs: &[Vec<f32>]| {
        ScalarQuantizer::from_vectors_with_sample_size(vs, 0.95, vs.len(), 7, 100).unwrap()
    };
    assert_eq!(read(&many), read(&whole));
    let auto = |vs: &[Vec<f32>]| {
        ScalarQuantizer::from_vectors_auto_interval(
            vs,
            VectorSimilarityFunction::DotProduct,
            vs.len(),
            7,
        )
        .unwrap()
    };
    assert_eq!(auto(&many), auto(&copies(1000)));
    let bad = vec![vec![f32::INFINITY; 16]; 30];
    assert_eq!(
        ScalarQuantizer::from_vectors_auto_interval(
            &bad,
            VectorSimilarityFunction::DotProduct,
            bad.len(),
            7,
        ),
        Err(QuantizerError::NonFiniteQuantiles)
    );
}

#[test]
fn quantile_selection() {
    let mut two = [3.0f32, 1.0];
    assert_eq!(upper_and_lower_quantile(&mut two, 0.9), (1.0, 3.0));
    let mut arr: Vec<f32> = (0..100).map(|i| ((i * 37) % 100) as f32).collect();
    // (100 * 0.1 / 2 + 0.5) = 5 dropped at each end.
    assert_eq!(upper_and_lower_quantile(&mut arr, 0.9), (5.0, 94.0));
    assert_eq!(java_float_compare(-0.0, 0.0), std::cmp::Ordering::Less);
    assert_eq!(
        java_float_compare(f32::NAN, f32::INFINITY),
        std::cmp::Ordering::Greater
    );
}

#[test]
fn quantized_similarities() {
    let e = ScalarQuantizedVectorSimilarity::from_vector_similarity(
        VectorSimilarityFunction::Euclidean,
        0.5,
        7,
    );
    assert_eq!(e.score(&[1, 2], 0.0, &[3, 2], 0.0), 1.0 / (1.0 + 2.0));
    let d = ScalarQuantizedVectorSimilarity::from_vector_similarity(
        VectorSimilarityFunction::Cosine,
        0.5,
        4,
    );
    assert_eq!(
        d.score(&[1, 2], 0.25, &[3, 2], 0.25),
        (1.0 + 3.5 + 0.5) / 2.0
    );
    let d8 = ScalarQuantizedVectorSimilarity::from_vector_similarity(
        VectorSimilarityFunction::DotProduct,
        0.5,
        8,
    );
    assert_eq!(d8.score(&[200], 0.0, &[2], 0.0), (1.0 + 200.0) / 2.0);
    let m = ScalarQuantizedVectorSimilarity::from_vector_similarity(
        VectorSimilarityFunction::MaximumInnerProduct,
        1.0,
        8,
    );
    assert_eq!(m.score(&[1], 0.0, &[2], 0.0), 3.0);
}

#[test]
fn top10_queue_keeps_the_best_and_pops_least_first() {
    let mut q = Vec::new();
    for i in 0..20usize {
        insert_top10(&mut q, (i, (i % 7) as f32));
    }
    assert_eq!(q.len(), 10);
    // Least first: lowest score, and among ties the highest doc.
    assert!(q.windows(2).all(|w| hit_less(w[0], w[1])));
    assert!(q.iter().all(|&(_, s)| s >= 3.0));
}
