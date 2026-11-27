#![cfg(test)]
// A test's `i + 1` is not a length read off disk; see
// `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

/// Rebuilds a file's footer after a test has patched a byte in its body.
/// Every corruption test needs this: `open` verifies the whole-file CRC
/// *before* it decodes any field, so a raw byte flip is rejected as a bad
/// checksum and never reaches the check under test.
fn repack(buf: &[u8], patch: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut body = buf[..buf.len() - codec_util::FOOTER_LENGTH].to_vec();
    patch(&mut body);
    codec_util::write_footer(&mut body);
    body
}

const ID: [u8; ID_LENGTH] = *b"vectorstestid001";

// ---------------- re-signed byte-flip sweep ----------------

/// A three-field segment that spans every structure the `.vemf`/`.vec`
/// pair has: a **dense** float field (no ord->doc structures at all), a
/// **sparse** float field (an `IndexedDISI` bitset *and* a
/// `DirectMonotonic` ordinal->doc sequence, both living in `.vec` after
/// the vectors), and a **byte** field (a different `byteSize`, and
/// therefore a different `alignOutput` padding). 600 documents, so the
/// DISI's dense-block path and its rank table are both populated -- a
/// single-document fixture leaves all three structures degenerate and
/// measures the fixture rather than the decoder.
fn sweep_fixture() -> (Vec<u8>, Vec<u8>) {
    let max_doc = 600;
    let dense: Vec<i32> = (0..max_doc).collect();
    let sparse: Vec<i32> = (0..max_doc).filter(|d| d % 3 == 0).collect();
    let byte_docs: Vec<i32> = (0..max_doc).filter(|d| d % 7 == 0).collect();
    let byte_values: Vec<u8> = (0..byte_docs.len() * 2).map(|i| (i % 251) as u8).collect();
    let fields = vec![
        float_field(0, 4, dense),
        float_field(1, 3, sparse),
        FlatVectorsField {
            field_number: 2,
            similarity: VectorSimilarityFunction::DotProduct,
            dimension: 2,
            docs: byte_docs,
            values: FieldVectorData::Byte(byte_values),
        },
    ];
    write(&fields, max_doc)
}

/// Everything a reader does with a `.vemf`/`.vec` pair, so that a flip
/// anywhere in either file has to surface as a typed error (or a clean,
/// self-consistent decode) rather than a panic or an allocation abort.
fn walk_everything(meta: &[u8], data: &[u8]) -> Result<()> {
    let reader = open(meta, data)?;
    for entry in reader.fields().to_vec() {
        let number = entry.field_number;
        match entry.encoding {
            VectorEncoding::Float32 => {
                let values = reader.float_vector_values(number)?;
                let mut scratch = vec![0.0f32; values.dimension()];
                for ord in 0..values.size() {
                    values.vector_into(ord, &mut scratch)?;
                    values.ord_to_doc(ord)?;
                }
                if values.size() > 1 {
                    values.raw_range(0, values.size())?;
                }
                let query = vec![0.25f32; values.dimension()];
                values.exhaustive_search(&query, 5)?;
                let mut cursor = values.doc_to_ord()?;
                for doc in 0..600 {
                    cursor.ordinal(doc)?;
                }
            }
            VectorEncoding::Byte => {
                let values = reader.byte_vector_values(number)?;
                for ord in 0..values.size() {
                    values.vector(ord)?;
                    values.ord_to_doc(ord)?;
                }
                let mut cursor = values.doc_to_ord()?;
                for doc in 0..600 {
                    cursor.ordinal(doc)?;
                }
            }
        }
    }
    Ok(())
}

/// Flip bit 0 and bit 7 of every `.vemf` and `.vec` body byte, re-sign the
/// footer so only a semantic invariant can reject the file, and require a
/// typed error or a clean decode from the full walk above. Never a panic,
/// never an abort.
///
/// A low rejection rate is expected and is not a gap: almost every `.vec`
/// byte is a vector *component*, and flipping one yields a different but
/// perfectly well-formed float. The bar is that nothing panics, nothing
/// aborts, and nothing reserves memory proportional to a number it just
/// read.
#[test]
fn every_resigned_single_byte_vemf_and_vec_corruption_is_an_error_or_a_clean_decode() {
    let (data, meta) = sweep_fixture();
    walk_everything(&meta, &data).expect("the fixture itself must decode");
    let mut flipped = 0usize;
    let mut rejected = 0usize;
    let mut meta_flipped = 0usize;
    let mut meta_rejected = 0usize;
    for which in 0..2 {
        let original: &[u8] = if which == 0 { &meta } else { &data };
        let body_len = original.len() - codec_util::FOOTER_LENGTH;
        for at in 0..body_len {
            for bit in [0u8, 7] {
                let patched = repack(original, |b| b[at] ^= 1 << bit);
                let (m, d) = if which == 0 {
                    (patched.as_slice(), data.as_slice())
                } else {
                    (meta.as_slice(), patched.as_slice())
                };
                flipped += 1;
                let bad = walk_everything(m, d).is_err();
                if bad {
                    rejected += 1;
                }
                if which == 0 {
                    meta_flipped += 1;
                    meta_rejected += usize::from(bad);
                }
            }
        }
    }
    assert_eq!(
        flipped,
        2 * (meta.len() + data.len() - 2 * codec_util::FOOTER_LENGTH)
    );
    // The metadata is all lengths, offsets and counts, so almost every
    // flip there has to be caught by something.
    assert!(
        meta_rejected > meta_flipped / 2,
        "only {meta_rejected} of {meta_flipped} .vemf flips rejected"
    );
    eprintln!(
        ".vemf+.vec byte-flip sweep: {rejected}/{flipped} rejected \
             (.vemf alone {meta_rejected}/{meta_flipped})"
    );
}

fn float_field(number: i32, dim: i32, docs: Vec<i32>) -> FlatVectorsField {
    let values = (0..docs.len() as i32 * dim)
        .map(|i| i as f32 * 0.5 - 1.0)
        .collect();
    FlatVectorsField {
        field_number: number,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension: dim,
        docs,
        values: FieldVectorData::Float32(values),
    }
}

fn write(fields: &[FlatVectorsField], max_doc: i32) -> (Vec<u8>, Vec<u8>) {
    write_flat_vectors(fields, max_doc, &ID, "").unwrap()
}

/// Position of a field's `dimension` vint in the `.vemf`, for the
/// corruption tests. The layout up to it is fixed-width apart from the two
/// vlongs, which are re-derived here rather than hard-coded.
fn open(meta: &[u8], data: &[u8]) -> Result<FlatVectorsReader<'static>> {
    // `FlatVectorsReader` borrows `data`; the tests only need the error, so
    // leak the buffer rather than thread lifetimes through every case.
    let data: &'static [u8] = Box::leak(data.to_vec().into_boxed_slice());
    FlatVectorsReader::open(meta, data, &ID, "")
}

// ---------------- kernels ----------------

fn naive_square_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x - *y) as f64).powi(2))
        .sum()
}

#[test]
fn float_kernels_agree_with_a_naive_reference_at_every_length() {
    // Lengths either side of the 8-, 16- and 32-lane splits, so every
    // remainder is exercised too.
    for len in [
        1usize, 5, 8, 9, 16, 17, 31, 32, 33, 40, 47, 64, 100, 128, 768,
    ] {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37 + 0.3).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.91 + 0.7).cos()).collect();
        let sq = square_distance(&a, &b) as f64;
        assert!(
            (sq - naive_square_distance(&a, &b)).abs() < 1e-5 * (len as f64),
            "len {len}: {sq}"
        );
        let dot = dot_product(&a, &b) as f64;
        let naive_dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        assert!(
            (dot - naive_dot).abs() < 1e-5 * (len as f64),
            "len {len}: {dot}"
        );
        let n1: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum();
        let n2: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum();
        let cos = cosine(&a, &b) as f64;
        assert!(
            (cos - naive_dot / (n1 * n2).sqrt()).abs() < 1e-4,
            "len {len}"
        );
    }
}

#[test]
fn cosine_returns_zero_for_a_zero_vector() {
    // Java returns NaN here (and asserts `Float.isFinite`); this port
    // refuses to poison the heap ordering with a NaN.
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    assert_eq!(cosine(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
    assert_eq!(cosine_bytes(&[0, 0], &[1, 2]), 0.0);
    assert_eq!(cosine_bytes(&[1, 2], &[0, 0]), 0.0);
}

#[test]
fn byte_kernels_sign_extend_like_javas_byte() {
    // 0xFF is -1, not 255: Java's `byte` is signed and the on-disk bytes
    // are those signed values.
    assert_eq!(dot_product_bytes(&[0xFF], &[1]), -1);
    assert_eq!(square_distance_bytes(&[0xFF], &[1]), 4);
    assert_eq!(dot_product_bytes(&[0x80], &[1]), -128);
    assert!((cosine_bytes(&[0xFF], &[1]) + 1.0).abs() < 1e-6);
}

#[test]
fn scale_max_inner_product_score_is_piecewise() {
    assert_eq!(scale_max_inner_product_score(0.0), 1.0);
    assert_eq!(scale_max_inner_product_score(3.0), 4.0);
    assert_eq!(scale_max_inner_product_score(-3.0), 0.25);
}

#[test]
fn normalize_helpers_match_java() {
    assert_eq!(normalize_to_unit_interval(1.0), 1.0);
    assert_eq!(normalize_to_unit_interval(-1.0), 0.0);
    // The `max(_, 0)` floor: an unnormalized dot below -1 must not score
    // negative.
    assert_eq!(normalize_to_unit_interval(-5.0), 0.0);
    assert_eq!(normalize_distance_to_unit_interval(0.0), 1.0);
    assert_eq!(normalize_distance_to_unit_interval(3.0), 0.25);
}

#[test]
fn float_similarity_transforms_match_java() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert_eq!(VectorSimilarityFunction::Euclidean.score(&a, &b), 1.0 / 3.0);
    assert_eq!(VectorSimilarityFunction::DotProduct.score(&a, &b), 0.5);
    assert_eq!(VectorSimilarityFunction::Cosine.score(&a, &b), 0.5);
    assert_eq!(
        VectorSimilarityFunction::MaximumInnerProduct.score(&a, &b),
        1.0
    );
    assert_eq!(VectorSimilarityFunction::DotProduct.score(&a, &a), 1.0);
    assert_eq!(
        VectorSimilarityFunction::MaximumInnerProduct.score(&a, &[-2.0, 0.0, 0.0]),
        1.0 / 3.0
    );
}

/// Byte vectors do **not** reuse the float transforms: DOT_PRODUCT is
/// `dotProductScore` (0.5 + dot / (dim * 2^15)), and COSINE has no
/// `max(_, 0)` clamp.
#[test]
fn byte_similarity_transforms_differ_from_the_float_ones() {
    let a = [1u8, 0, 0];
    let b = [0u8, 1, 0];
    assert_eq!(
        VectorSimilarityFunction::Euclidean.score_bytes(&a, &b),
        1.0 / 3.0
    );
    assert_eq!(
        VectorSimilarityFunction::DotProduct.score_bytes(&a, &b),
        0.5
    );
    // 3 dims, dot = 1 -> 0.5 + 1 / (3 * 32768)
    assert_eq!(
        VectorSimilarityFunction::DotProduct.score_bytes(&a, &a),
        0.5 + 1.0 / (3.0 * 32768.0)
    );
    assert_eq!(VectorSimilarityFunction::Cosine.score_bytes(&a, &b), 0.5);
    assert_eq!(
        VectorSimilarityFunction::MaximumInnerProduct.score_bytes(&a, &a),
        2.0
    );
}

// ---------------- writer / reader ----------------

#[test]
fn empty_field_writes_the_minus_two_marker() {
    let field = FlatVectorsField {
        field_number: 4,
        similarity: VectorSimilarityFunction::Cosine,
        dimension: 3,
        docs: Vec::new(),
        values: FieldVectorData::Float32(Vec::new()),
    };
    let (data, meta) = write(&[field], 10);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let entry = reader.field(4).unwrap();
    assert!(entry.ord_to_doc.is_empty());
    assert!(!entry.ord_to_doc.is_dense());
    let values = reader.float_vector_values(4).unwrap();
    assert_eq!(values.size(), 0);
    assert!(values.is_empty());
    assert!(values
        .exhaustive_search(&[0.0, 0.0, 0.0], 3)
        .unwrap()
        .is_empty());
    let mut cursor = values.doc_to_ord().unwrap();
    assert_eq!(cursor.ordinal(0).unwrap(), None);
    cursor.reset();
    assert_eq!(cursor.ordinal(9).unwrap(), None);
}

#[test]
fn dense_cursor_answers_out_of_range_docs_with_none() {
    let (data, meta) = write(&[float_field(0, 2, (0..5).collect())], 5);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = reader.float_vector_values(0).unwrap();
    let mut cursor = values.doc_to_ord().unwrap();
    assert_eq!(cursor.ordinal(4).unwrap(), Some(4));
    assert_eq!(cursor.ordinal(5).unwrap(), None);
    cursor.reset();
    assert_eq!(cursor.ordinal(0).unwrap(), Some(0));
}

#[test]
fn exhaustive_search_ranks_by_score_then_doc_id() {
    // Two vectors equidistant from the query: the smaller doc id wins.
    let field = FlatVectorsField {
        field_number: 0,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension: 1,
        docs: vec![0, 1, 2],
        values: FieldVectorData::Float32(vec![-1.0, 1.0, 5.0]),
    };
    let (data, meta) = write(&[field], 3);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = reader.float_vector_values(0).unwrap();
    let hits = values.exhaustive_search(&[0.0], 3).unwrap();
    assert_eq!(hits[0].0, 0);
    assert_eq!(hits[1].0, 1);
    assert_eq!(hits[2].0, 2);
    assert_eq!(values.exhaustive_search(&[0.0], 0).unwrap(), Vec::new());
    assert_eq!(values.exhaustive_search(&[0.0], 2).unwrap().len(), 2);
}

#[test]
fn byte_exhaustive_search_and_scorer() {
    let field = FlatVectorsField {
        field_number: 1,
        similarity: VectorSimilarityFunction::Cosine,
        dimension: 2,
        docs: vec![0, 2],
        values: FieldVectorData::Byte(vec![1, 0, 0, 1]),
    };
    let (data, meta) = write(&[field], 4);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = reader.byte_vector_values(1).unwrap();
    assert_eq!(values.dimension(), 2);
    assert_eq!(values.similarity(), VectorSimilarityFunction::Cosine);
    assert!(!values.is_empty());
    let hits = values.exhaustive_search(&[1, 0], 2).unwrap();
    assert_eq!(hits[0].0, 0);
    assert_eq!(hits[1].0, 2);
    assert_eq!(values.exhaustive_search(&[1, 0], 0).unwrap(), Vec::new());
    assert!(matches!(
        values.exhaustive_search(&[1], 1),
        Err(Error::QueryDimensionMismatch(2, 1))
    ));
    assert!(matches!(
        values.scorer(&[1]),
        Err(Error::QueryDimensionMismatch(2, 1))
    ));
    assert!(values.scorer(&[1, 2]).is_ok());
}

#[test]
fn search_rejects_a_query_of_the_wrong_dimension() {
    let (data, meta) = write(&[float_field(0, 4, (0..3).collect())], 3);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = reader.float_vector_values(0).unwrap();
    assert!(matches!(
        values.exhaustive_search(&[1.0, 2.0], 1),
        Err(Error::QueryDimensionMismatch(4, 2))
    ));
    assert!(matches!(
        values.scorer(&[1.0, 2.0]),
        Err(Error::QueryDimensionMismatch(4, 2))
    ));
}

#[test]
fn unknown_field_and_encoding_mismatch_are_rejected() {
    let byte_field = FlatVectorsField {
        field_number: 2,
        similarity: VectorSimilarityFunction::DotProduct,
        dimension: 2,
        docs: vec![0, 1],
        values: FieldVectorData::Byte(vec![1, 2, 3, 4]),
    };
    let (data, meta) = write(&[float_field(0, 2, vec![0, 1]), byte_field], 2);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    assert!(matches!(
        reader.float_vector_values(9),
        Err(Error::UnknownField(9))
    ));
    assert!(matches!(
        reader.float_vector_values(2),
        Err(Error::EncodingMismatch(
            2,
            VectorEncoding::Byte,
            VectorEncoding::Float32
        ))
    ));
    assert!(matches!(
        reader.byte_vector_values(0),
        Err(Error::EncodingMismatch(
            0,
            VectorEncoding::Float32,
            VectorEncoding::Byte
        ))
    ));
    assert!(reader.field(9).is_none());
    assert_eq!(reader.fields().len(), 2);
}

#[test]
fn vector_into_rejects_a_wrong_sized_buffer() {
    let (data, meta) = write(&[float_field(0, 4, vec![0, 1])], 2);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = reader.float_vector_values(0).unwrap();
    let mut short = [0.0f32; 3];
    assert!(matches!(
        values.vector_into(0, &mut short),
        Err(Error::QueryDimensionMismatch(4, 3))
    ));
    let mut long = [0.0f32; 5];
    assert!(matches!(
        values.vector_into(0, &mut long),
        Err(Error::QueryDimensionMismatch(4, 5))
    ));
}

#[test]
fn out_of_range_ordinals_are_rejected() {
    let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = reader.float_vector_values(0).unwrap();
    assert!(matches!(values.vector(2), Err(Error::OrdOutOfRange(2, 2))));
    assert!(matches!(
        values.vector(-1),
        Err(Error::OrdOutOfRange(-1, 2))
    ));
    assert!(matches!(
        values.ord_to_doc(2),
        Err(Error::OrdOutOfRange(2, 2))
    ));
}

#[test]
fn sparse_ord_to_doc_rejects_an_out_of_range_ordinal() {
    let (data, meta) = write(&[float_field(0, 2, vec![0, 3])], 8);
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = reader.float_vector_values(0).unwrap();
    assert_eq!(values.ord_to_doc(1).unwrap(), 3);
    assert!(matches!(
        values.ord_to_doc(5),
        Err(Error::OrdOutOfRange(5, 2))
    ));
}

#[test]
fn bulk_ord_to_docs_matches_one_lookup_per_ordinal() {
    for (docs, max_doc) in [(vec![0, 1, 2], 3), (vec![1, 4, 5, 9], 12)] {
        let (data, meta) = write(&[float_field(0, 2, docs.clone())], max_doc);
        let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
        let values = reader.float_vector_values(0).unwrap();
        let mut out = vec![-7];
        values.ord_to_docs(&mut out).unwrap();
        let one_by_one: Vec<i32> = (0..values.size())
            .map(|ord| values.ord_to_doc(ord).unwrap())
            .collect();
        assert_eq!(out, one_by_one);
        assert_eq!(out, docs);
    }
    let explicit = RawVectorValues {
        slice: &[],
        file: &[],
        dimension: 1,
        vector_bytes: 4,
        size: 3,
        similarity: VectorSimilarityFunction::Euclidean,
        hamming: false,
        half: false,
        ord_to_doc: OrdToDoc::Explicit(std::sync::Arc::from(vec![2, 5, 9])),
    };
    let mut out = Vec::new();
    explicit.ord_to_docs(&mut out).unwrap();
    assert_eq!(out, [2, 5, 9]);
}

// ---------------- half floats ----------------

const HALF: &str = "KNN1040HnswHalfFloatVectorsFormat_0";

#[test]
fn half_float_conversion_matches_java() {
    // Values and `Float.floatToFloat16` results from a JDK 25 jshell.
    for (f, h) in [
        (0.0f32, 0x0000u16),
        (-0.0, 0x8000),
        (1.0, 0x3c00),
        (-2.5, 0xc100),
        (65504.0, 0x7bff),
        (65519.0, 0x7bff),
        (65520.0, 0x7c00),
        (f32::INFINITY, 0x7c00),
        (6.1035156e-5, 0x0400),
        (5.9604645e-8, 0x0001),
        (2.9802322e-8, 0x0000),
        (2.9802326e-8, 0x0001),
        (0.1, 0x2e66),
        (1.0009766, 0x3c01),
        (1.0004883, 0x3c00),
        (1.0014648, 0x3c02),
        (2047.5, 0x6800),
        (1e-6, 0x0011),
        (-3.3333, 0xc2ab),
    ] {
        assert_eq!(f32_to_f16(f), h, "{f}");
    }
    assert_eq!(f32_to_f16(f32::NAN), 0x7e00);
    // Every 251st f32 bit pattern, NaNs included, hashed as
    // `sum = sum * 31 + floatToFloat16(f)` in Java.
    let mut sum = 0u64;
    for i in 0..=(u32::MAX / 251) {
        let h = f32_to_f16(f32::from_bits(i * 251));
        sum = sum.wrapping_mul(31).wrapping_add(u64::from(h));
    }
    assert_eq!(sum, 0x4a56_b634_55f0_81d9);
    // Every half float survives a round trip.
    for h in 0..=u16::MAX {
        let f = f16_to_f32(h);
        if f.is_nan() {
            continue;
        }
        assert_eq!(f32_to_f16(f), h, "{h:#x}");
    }
}

fn open_half(meta: &[u8], data: &[u8]) -> Result<FlatVectorsReader<'static>> {
    let data: &'static [u8] = Box::leak(data.to_vec().into_boxed_slice());
    FlatVectorsReader::open(meta, data, &ID, HALF)
}

#[test]
fn half_float_fields_round_trip_and_need_their_similarity() {
    let mut field = float_field(3, 4, vec![0, 2, 5]);
    field.similarity = VectorSimilarityFunction::DotProduct;
    let (data, meta) = write_flat_vectors(&[field], 6, &ID, HALF).unwrap();
    // Two bytes a component, after a 64-aligned header.
    let mut reader = open_half(&meta, &data).unwrap();
    assert!(reader.stores_half_floats());
    let entry = reader.field(3).unwrap();
    assert_eq!(entry.vector_data_offset % 4096, 0);
    assert_eq!(entry.vector_data_length, 3 * 4 * 2);
    assert!(reader.float_vector_values(3).is_err());
    reader
        .resolve_similarities(|n| (n == 3).then_some(VectorSimilarityFunction::Cosine))
        .unwrap();
    assert_eq!(
        reader.field(3).unwrap().similarity,
        VectorSimilarityFunction::Cosine
    );
    // Scored as k-NN's FP16 cosine kernel scores normalized vectors.
    let values = reader.float_vector_values(3).unwrap();
    assert_eq!(values.similarity(), VectorSimilarityFunction::DotProduct);
    assert_eq!(values.ord_to_doc(1).unwrap(), 2);
    // `float_field`'s components (`i * 0.5 - 1`) are exact half floats.
    let want: Vec<f32> = (4..8).map(|i| i as f32 * 0.5 - 1.0).collect();
    assert_eq!(values.vector(1).unwrap(), want);
    let mut scratch = vec![0.0; 4];
    assert_eq!(values.vector_ref(1, &mut scratch).unwrap(), want);
    // Half floats are decoded, never borrowed.
    let all = values.all_vectors().unwrap();
    assert!(matches!(all, std::borrow::Cow::Owned(_)));
    let every: Vec<f32> = (0..3).flat_map(|o| values.vector(o).unwrap()).collect();
    assert_eq!(*all, every[..]);
    // The stock files are not the Lucene99 ones.
    assert!(open(&meta, &data).is_err());
}

#[test]
fn a_merge_converts_between_full_and_half_floats() {
    let full = float_field(1, 3, vec![0, 1]);
    let (full_data, full_meta) = write(&[full], 2);
    let full_reader = open(&full_meta, &full_data).unwrap();
    let (half_data, half_meta) =
        write_flat_vectors(&[float_field(1, 3, vec![0, 1])], 2, &ID, HALF).unwrap();
    let mut half_reader = open_half(&half_meta, &half_data).unwrap();
    half_reader
        .resolve_similarities(|_| Some(VectorSimilarityFunction::Euclidean))
        .unwrap();
    let map = [0, 1];
    let map2 = [2, 3];
    let sources = [
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(full_reader.float_vector_values(1).unwrap()),
            doc_map: &map,
        },
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(half_reader.float_vector_values(1).unwrap()),
            doc_map: &map2,
        },
    ];
    for suffix in ["", HALF] {
        let mut writer = FlatVectorsWriter::new(4, &ID, suffix);
        writer
            .merge_one_flat_vector_field(&MergedFlatVectorField {
                field_number: 1,
                encoding: VectorEncoding::Float32,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 3,
                sources: &sources,
            })
            .unwrap();
        let (data, meta) = writer.finish();
        let data: &'static [u8] = Box::leak(data.into_boxed_slice());
        let mut merged = FlatVectorsReader::open(&meta, data, &ID, suffix).unwrap();
        merged
            .resolve_similarities(|_| Some(VectorSimilarityFunction::Euclidean))
            .unwrap();
        let values = merged.float_vector_values(1).unwrap();
        for ord in 0..4 {
            let want: Vec<f32> = (0..3)
                .map(|i| (i + (ord % 2) * 3) as f32 * 0.5 - 1.0)
                .collect();
            assert_eq!(values.vector(ord).unwrap(), want, "{suffix} ord {ord}");
        }
    }
}

#[test]
fn a_streamed_merge_writes_the_bytes_a_buffered_one_does() {
    let full = float_field(1, 3, vec![0, 1]);
    let (full_data, full_meta) = write(&[full], 2);
    let full_reader = open(&full_meta, &full_data).unwrap();
    let sources = [
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(full_reader.float_vector_values(1).unwrap()),
            doc_map: &[],
        },
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(full_reader.float_vector_values(1).unwrap()),
            doc_map: &[],
        },
    ];
    let dir = std::env::temp_dir().join(format!("streamed-flat-merge-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // A sparse merged field (4 of 12 documents) and a dense one (4 of 4).
    for (max_doc, map_a, map_b) in [(12, [3, 5], [6, 9]), (4, [0, 1], [2, 3])] {
        let sources = [
            FlatVectorMergeSource {
                doc_map: &map_a,
                values: match &sources[0].values {
                    MergeSourceValues::Float32(v) => MergeSourceValues::Float32(v.clone()),
                    _ => unreachable!(),
                },
            },
            FlatVectorMergeSource {
                doc_map: &map_b,
                values: match &sources[1].values {
                    MergeSourceValues::Float32(v) => MergeSourceValues::Float32(v.clone()),
                    _ => unreachable!(),
                },
            },
        ];
        let field = MergedFlatVectorField {
            field_number: 1,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 3,
            sources: &sources,
        };
        let mut buffered = FlatVectorsWriter::new(max_doc, &ID, "");
        buffered.merge_one_flat_vector_field(&field).unwrap();
        let (want_data, want_meta) = buffered.finish();

        let mut streamed = FlatVectorsWriter::new(max_doc, &ID, "");
        let out = lucene_store::FsIndexOutput::create(&dir, "out.vec").unwrap();
        streamed.stream_to(out);
        streamed.merge_one_flat_vector_field(&field).unwrap();
        let (out, meta) = streamed.finish_streamed();
        out.unwrap().close().unwrap();
        assert_eq!(std::fs::read(dir.join("out.vec")).unwrap(), want_data);
        assert_eq!(meta, want_meta);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------- write-side validation ----------------

#[test]
fn writer_rejects_malformed_fields() {
    let mut bad = float_field(0, 0, vec![0]);
    bad.values = FieldVectorData::Float32(Vec::new());
    assert!(matches!(
        write_flat_vectors(&[bad], 1, &ID, ""),
        Err(Error::DimensionMismatch(0, 0, 0))
    ));

    let mut wrong_len = float_field(0, 2, vec![0, 1]);
    wrong_len.values = FieldVectorData::Float32(vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        write_flat_vectors(&[wrong_len], 2, &ID, ""),
        Err(Error::DimensionMismatch(0, 4, 3))
    ));

    let descending = float_field(0, 1, vec![3, 1]);
    assert!(matches!(
        write_flat_vectors(&[descending], 5, &ID, ""),
        Err(Error::CorruptMeta(_))
    ));

    let past_max_doc = float_field(0, 1, vec![0, 9]);
    assert!(matches!(
        write_flat_vectors(&[past_max_doc], 5, &ID, ""),
        Err(Error::CorruptMeta(_))
    ));

    let negative = float_field(0, 1, vec![-1, 2]);
    assert!(matches!(
        write_flat_vectors(&[negative], 5, &ID, ""),
        Err(Error::CorruptMeta(_))
    ));

    let mut orphan = float_field(0, 2, Vec::new());
    orphan.values = FieldVectorData::Float32(vec![1.0, 2.0]);
    assert!(matches!(
        write_flat_vectors(&[orphan], 5, &ID, ""),
        Err(Error::DimensionMismatch(0, 0, 2))
    ));
}

/// `VectorUtil.checkFinite`, which `KnnFloatVectorField`'s constructor runs
/// on every indexed vector -- so a Lucene-written `.vec` never contains one
/// of these and this port's must not either.
#[test]
fn writer_rejects_non_finite_components() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let field = FlatVectorsField {
            field_number: 2,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            docs: vec![0, 1],
            values: FieldVectorData::Float32(vec![1.0, 2.0, bad, 4.0]),
        };
        assert!(
            matches!(
                write_flat_vectors(&[field], 2, &ID, ""),
                Err(Error::NonFiniteValue(2, 2, _))
            ),
            "{bad} accepted"
        );
    }
    // A byte field has no non-finite values to reject.
    let bytes = FlatVectorsField {
        field_number: 2,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension: 2,
        docs: vec![0, 1],
        values: FieldVectorData::Byte(vec![1, 2, 3, 4]),
    };
    assert!(write_flat_vectors(&[bytes], 2, &ID, "").is_ok());
}

#[test]
fn field_vector_data_reports_its_encoding() {
    assert_eq!(
        FieldVectorData::Float32(Vec::new()).encoding(),
        VectorEncoding::Float32
    );
    assert_eq!(
        FieldVectorData::Byte(Vec::new()).encoding(),
        VectorEncoding::Byte
    );
}

// ---------------- corrupt metadata ----------------

#[test]
fn version_mismatch_between_meta_and_data_is_rejected() {
    let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
    // The version is the 4 bytes right after CODEC_MAGIC + the codec name.
    let version_at = codec_util::header_length(DATA_CODEC) - 4;
    let bumped = repack(&data, |b| {
        b[version_at..version_at + 4].copy_from_slice(&1i32.to_be_bytes());
    });
    // A version outside [VERSION_START, VERSION_CURRENT] is caught by
    // `check_index_header` itself, which is the same rejection.
    assert!(open(&meta, &bumped).is_err());
}

/// Corrupts the first field's encoding/similarity ordinal, which sit at a
/// fixed offset right after the `.vemf` header's field number.
fn meta_field_start() -> usize {
    codec_util::index_header_length(META_CODEC, "") + 4
}

#[test]
fn invalid_encoding_and_similarity_ordinals_are_rejected() {
    let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
    let at = meta_field_start();
    let bad_encoding = repack(&meta, |b| {
        b[at..at + 4].copy_from_slice(&7i32.to_le_bytes());
    });
    assert!(matches!(
        open(&bad_encoding, &data),
        Err(Error::CorruptMeta(_))
    ));
    let bad_similarity = repack(&meta, |b| {
        b[at + 4..at + 8].copy_from_slice(&9i32.to_le_bytes());
    });
    assert!(matches!(
        open(&bad_similarity, &data),
        Err(Error::CorruptMeta(_))
    ));
}

#[test]
fn a_negative_field_number_is_rejected() {
    let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
    let at = codec_util::index_header_length(META_CODEC, "");
    let bad = repack(&meta, |b| {
        b[at..at + 4].copy_from_slice(&(-5i32).to_le_bytes());
    });
    assert!(matches!(open(&bad, &data), Err(Error::CorruptMeta(_))));
}

#[test]
fn a_vector_data_length_that_disagrees_with_size_is_rejected() {
    // `dimension` is a vint at a known offset: header, fieldNumber(4),
    // encoding(4), similarity(4), then two vlongs. Both vlongs here are
    // small enough to be one or two bytes, so find the dimension by
    // decoding rather than guessing.
    let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
    let mut input = SliceInput::new(&meta);
    codec_util::check_index_header(
        &mut input,
        META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        &ID,
        "",
    )
    .unwrap();
    input.read_i32().unwrap(); // field number
    input.read_i32().unwrap(); // encoding
    input.read_i32().unwrap(); // similarity
    input.read_vlong().unwrap(); // data offset
    input.read_vlong().unwrap(); // data length
    let dim_at = input.position();
    assert_eq!(meta[dim_at], 2); // a one-byte vint
    let bad = repack(&meta, |b| b[dim_at] = 3);
    assert!(matches!(open(&bad, &data), Err(Error::CorruptMeta(_))));
    // ... and a dimension of zero, which Java's `FieldEntry` would divide
    // by.
    let zero = repack(&meta, |b| b[dim_at] = 0);
    assert!(matches!(open(&zero, &data), Err(Error::CorruptMeta(_))));
    // ... and a negative count.
    let count_at = dim_at + 1;
    let negative_count = repack(&meta, |b| {
        b[count_at..count_at + 4].copy_from_slice(&(-1i32).to_le_bytes());
    });
    assert!(matches!(
        open(&negative_count, &data),
        Err(Error::CorruptMeta(_))
    ));
    // ... and a `docsWithFieldOffset` that is neither -1, -2 nor a real
    // offset.
    let offset_at = count_at + 4;
    let bad_marker = repack(&meta, |b| {
        b[offset_at..offset_at + 8].copy_from_slice(&(-3i64).to_le_bytes());
    });
    assert!(matches!(
        open(&bad_marker, &data),
        Err(Error::CorruptMeta(_))
    ));
}

#[test]
fn a_vector_region_past_the_end_of_the_data_file_is_rejected() {
    let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
    let mut input = SliceInput::new(&meta);
    codec_util::check_index_header(
        &mut input,
        META_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        &ID,
        "",
    )
    .unwrap();
    input.read_i32().unwrap();
    input.read_i32().unwrap();
    input.read_i32().unwrap();
    let offset_at = input.position();
    // The offset is a vlong; the writer's is 64 (one byte can't hold it,
    // so it is two). Overwrite with a huge two-byte vlong.
    let bad = repack(&meta, |b| {
        b[offset_at] = 0xFF;
        b[offset_at + 1] = 0x7F;
    });
    assert!(matches!(open(&bad, &data), Err(Error::CorruptMeta(_))));
}

#[test]
fn a_truncated_footer_is_rejected() {
    let (data, meta) = write(&[float_field(0, 2, vec![0, 1])], 2);
    assert!(open(&meta[..meta.len() - 1], &data).is_err());
}
// ---------------- merge ----------------

/// Two source segments, written as this port writes any segment, reopened
/// and merged. The merged files must be **byte for byte** what a single
/// flush of the same documents in the same order produces -- which is the
/// whole contract: a merge may not be observable in the output.
#[test]
fn a_merge_produces_exactly_what_one_flush_of_the_same_documents_would() {
    // Segment A: 5 documents, dense. Segment B: 4 documents, sparse
    // (docs 0 and 2 only).
    let a_docs: Vec<i32> = (0..5).collect();
    let b_docs: Vec<i32> = vec![0, 2];
    let dim = 3;
    let a_values: Vec<f32> = (0..5 * dim).map(|i| i as f32 * 0.25 - 2.0).collect();
    let b_values: Vec<f32> = (0..2 * dim).map(|i| i as f32 * -0.75 + 9.0).collect();

    let (a_vec, a_meta) = write(
        &[FlatVectorsField {
            field_number: 7,
            similarity: VectorSimilarityFunction::Cosine,
            dimension: dim,
            docs: a_docs.clone(),
            values: FieldVectorData::Float32(a_values.clone()),
        }],
        5,
    );
    let (b_vec, b_meta) = write(
        &[FlatVectorsField {
            field_number: 7,
            similarity: VectorSimilarityFunction::Cosine,
            dimension: dim,
            docs: b_docs.clone(),
            values: FieldVectorData::Float32(b_values.clone()),
        }],
        4,
    );

    let a = FlatVectorsReader::open(&a_meta, &a_vec, &ID, "").unwrap();
    let b = FlatVectorsReader::open(&b_meta, &b_vec, &ID, "").unwrap();
    // Nothing deleted: A's docs keep their ids, B's are appended after
    // them. This is the shape `merge.rs` produces for a plain merge.
    let a_map: Vec<i32> = (0..5).collect();
    let b_map: Vec<i32> = (0..4).map(|d| d + 5).collect();
    let sources = vec![
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(a.float_vector_values(7).unwrap()),
            doc_map: &a_map,
        },
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(b.float_vector_values(7).unwrap()),
            doc_map: &b_map,
        },
    ];
    let mut writer = FlatVectorsWriter::new(9, &ID, "");
    writer
        .merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 7,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Cosine,
            dimension: dim,
            sources: &sources,
        })
        .unwrap();
    let (merged_vec, merged_meta) = writer.finish();

    // The same documents, flushed in one go.
    let mut all_docs = a_docs.clone();
    all_docs.extend(b_docs.iter().map(|d| d + 5));
    let mut all_values = a_values.clone();
    all_values.extend_from_slice(&b_values);
    let (flush_vec, flush_meta) = write(
        &[FlatVectorsField {
            field_number: 7,
            similarity: VectorSimilarityFunction::Cosine,
            dimension: dim,
            docs: all_docs,
            values: FieldVectorData::Float32(all_values),
        }],
        9,
    );
    assert_eq!(
        merged_vec, flush_vec,
        ".vec differs between merge and flush"
    );
    assert_eq!(
        merged_meta, flush_meta,
        ".vemf differs between merge and flush"
    );
}

/// A document the merge drops takes its vector with it, and the ordinals
/// of everything after it shift down -- which is the part a bulk byte copy
/// can get wrong, since the surviving run is no longer the whole field.
#[test]
fn a_merge_drops_deleted_documents_and_renumbers_the_rest() {
    let dim = 2;
    let values: Vec<f32> = (0..6 * dim).map(|i| i as f32).collect();
    let (src_vec, src_meta) = write(
        &[FlatVectorsField {
            field_number: 1,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: dim,
            docs: (0..6).collect(),
            values: FieldVectorData::Float32(values.clone()),
        }],
        6,
    );
    let src = FlatVectorsReader::open(&src_meta, &src_vec, &ID, "").unwrap();
    // Drop docs 1 and 4 -- one inside a run, one at the end but not last,
    // so both the "flush the run" and "start a new run" branches fire.
    let map: Vec<i32> = vec![0, -1, 1, 2, -1, 3];
    let sources = vec![FlatVectorMergeSource {
        values: MergeSourceValues::Float32(src.float_vector_values(1).unwrap()),
        doc_map: &map,
    }];
    let mut writer = FlatVectorsWriter::new(4, &ID, "");
    writer
        .merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 1,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: dim,
            sources: &sources,
        })
        .unwrap();
    let (merged_vec, merged_meta) = writer.finish();

    let merged = FlatVectorsReader::open(&merged_meta, &merged_vec, &ID, "").unwrap();
    let out = merged.float_vector_values(1).unwrap();
    assert_eq!(out.size(), 4);
    // Dense again: four vectors over four documents.
    for (new_ord, old_doc) in [0i32, 2, 3, 5].into_iter().enumerate() {
        let new_ord = new_ord as i32;
        assert_eq!(out.ord_to_doc(new_ord).unwrap(), new_ord);
        assert_eq!(
            out.vector(new_ord).unwrap(),
            values[old_doc as usize * dim as usize..(old_doc as usize + 1) * dim as usize].to_vec()
        );
    }
}

/// A BYTE field merges the same way, and the encoding/dimension the caller
/// declares is cross-checked against every source rather than trusted --
/// a source opened for the wrong field would otherwise be copied in
/// verbatim and produce a `.vec` whose length no longer matches its
/// `.vemf`.
#[test]
fn merge_checks_every_source_against_the_declared_field() {
    let (f32_vec, f32_meta) = write(&[float_field(0, 4, (0..3).collect())], 3);
    let f32_reader = FlatVectorsReader::open(&f32_meta, &f32_vec, &ID, "").unwrap();
    let map: Vec<i32> = (0..3).collect();

    let sources = vec![FlatVectorMergeSource {
        values: MergeSourceValues::Float32(f32_reader.float_vector_values(0).unwrap()),
        doc_map: &map,
    }];
    let mut writer = FlatVectorsWriter::new(3, &ID, "");
    let clean = writer.data.len();
    // Declared BYTE, source is FLOAT32.
    assert!(matches!(
        writer.merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 0,
            encoding: VectorEncoding::Byte,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 4,
            sources: &sources,
        }),
        Err(Error::EncodingMismatch(
            0,
            VectorEncoding::Float32,
            VectorEncoding::Byte
        ))
    ));
    // Declared dimension disagrees with the source's.
    assert!(matches!(
        writer.merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 8,
            sources: &sources,
        }),
        Err(Error::DimensionMismatch(0, 8, 4))
    ));
    // A non-positive dimension.
    assert!(matches!(
        writer.merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 0,
            sources: &sources,
        }),
        Err(Error::DimensionMismatch(0, 0, 0))
    ));
    // A failed call must leave the writer untouched -- otherwise the
    // alignment padding (and, for a later failure, whole runs of copied
    // vectors) would leak into whatever field is written next.
    assert_eq!(writer.data.len(), clean);
    writer
        .merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 4,
            sources: &sources,
        })
        .unwrap();
    let (data, meta) = writer.finish();
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    assert_eq!(reader.float_vector_values(0).unwrap().size(), 3);
}

/// An **index-sorted** merge interleaves its sources: source 0's vectors
/// land on merged docs 0 and 3, source 1's on 1 and 2. The merged ordinal
/// space must then follow the merged *document* order, not source order --
/// `.vemf`'s doc list is an `IndexedDISI`, which only encodes a strictly
/// ascending list, so source order here is not merely a different
/// convention but an unwritable one.
///
/// The vectors themselves are checked component by component against the
/// document they came from: that is the assertion a wrong ordinal
/// assignment breaks while leaving a segment that decodes cleanly.
#[test]
fn an_interleaving_doc_map_assigns_merged_ordinals_in_document_order() {
    // Two sources of two vectors each, dimension 2, with distinguishable
    // values.
    let (vec_a, meta_a) = write(&[float_field(0, 2, vec![0, 1])], 2);
    let (vec_b, meta_b) = write(&[float_field(0, 2, vec![0, 1])], 2);
    let src_a = FlatVectorsReader::open(&meta_a, &vec_a, &ID, "").unwrap();
    let src_b = FlatVectorsReader::open(&meta_b, &vec_b, &ID, "").unwrap();
    let values_a = src_a.float_vector_values(0).unwrap();
    let values_b = src_b.float_vector_values(0).unwrap();
    let want_a: Vec<Vec<f32>> = (0..2)
        .map(|o| values_a.vector(o).unwrap().to_vec())
        .collect();
    let want_b: Vec<Vec<f32>> = (0..2)
        .map(|o| values_b.vector(o).unwrap().to_vec())
        .collect();

    // source 0 doc 0 -> merged 0, doc 1 -> merged 3;
    // source 1 doc 0 -> merged 1, doc 1 -> merged 2.
    let map_a = vec![0i32, 3];
    let map_b = vec![1i32, 2];
    let sources = vec![
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(values_a),
            doc_map: &map_a,
        },
        FlatVectorMergeSource {
            values: MergeSourceValues::Float32(values_b),
            doc_map: &map_b,
        },
    ];
    let mut writer = FlatVectorsWriter::new(4, &ID, "");
    writer
        .merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            sources: &sources,
        })
        .unwrap();
    let (data, meta) = writer.finish();
    let merged = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = merged.float_vector_values(0).unwrap();
    assert_eq!(values.size(), 4);
    let expected: Vec<(i32, &Vec<f32>)> = vec![
        (0, &want_a[0]),
        (1, &want_b[0]),
        (2, &want_b[1]),
        (3, &want_a[1]),
    ];
    for (ord, (doc, want)) in expected.iter().enumerate() {
        assert_eq!(values.ord_to_doc(ord as i32).unwrap(), *doc);
        assert_eq!(
            values.vector(ord as i32).unwrap(),
            want.as_slice(),
            "merged ordinal {ord} holds another document's vector"
        );
    }
}

/// A doc map that maps two source documents onto the same merged id, or
/// that runs backwards, produces a doc list `IndexedDISI.writeBitSet`
/// cannot encode -- and would otherwise yield a segment that decodes
/// cleanly and answers wrongly. Same rule the flush path enforces.
#[test]
fn merge_rejects_a_doc_map_that_is_not_ascending_or_is_short() {
    let (src_vec, src_meta) = write(&[float_field(0, 2, (0..3).collect())], 3);
    let src = FlatVectorsReader::open(&src_meta, &src_vec, &ID, "").unwrap();

    for map in [vec![2i32, 1, 0], vec![0, 0, 1]] {
        let sources = vec![FlatVectorMergeSource {
            values: MergeSourceValues::Float32(src.float_vector_values(0).unwrap()),
            doc_map: &map,
        }];
        let mut writer = FlatVectorsWriter::new(3, &ID, "");
        assert!(
            matches!(
                writer.merge_one_flat_vector_field(&MergedFlatVectorField {
                    field_number: 0,
                    encoding: VectorEncoding::Float32,
                    similarity: VectorSimilarityFunction::Euclidean,
                    dimension: 2,
                    sources: &sources,
                }),
                Err(Error::CorruptMeta(_))
            ),
            "doc map {map:?} must be rejected"
        );
    }

    // A doc map shorter than the source's document count.
    let short: Vec<i32> = vec![0];
    let sources = vec![FlatVectorMergeSource {
        values: MergeSourceValues::Float32(src.float_vector_values(0).unwrap()),
        doc_map: &short,
    }];
    let mut writer = FlatVectorsWriter::new(3, &ID, "");
    assert!(matches!(
        writer.merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            sources: &sources,
        }),
        Err(Error::CorruptMeta(_))
    ));
}

/// A field with no vectors in any source is legal: Java's
/// `DocsWithFieldSet` is simply empty and `writeStoredMeta` records the
/// `-2` empty marker.
#[test]
fn merging_a_field_no_source_has_vectors_for_writes_an_empty_field() {
    let mut writer = FlatVectorsWriter::new(10, &ID, "");
    writer
        .merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 3,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::DotProduct,
            dimension: 5,
            sources: &[],
        })
        .unwrap();
    let (data, meta) = writer.finish();
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    let values = reader.float_vector_values(3).unwrap();
    assert_eq!(values.size(), 0);
    assert!(values.is_empty());
    assert!(reader.field(3).unwrap().ord_to_doc.is_empty());
}

/// The writer can carry flushed and merged fields in one file pair, which
/// is what a merge that gains a brand-new field needs.
#[test]
fn a_flushed_field_and_a_merged_field_share_one_file_pair() {
    let (src_vec, src_meta) = write(&[float_field(0, 2, (0..4).collect())], 4);
    let src = FlatVectorsReader::open(&src_meta, &src_vec, &ID, "").unwrap();
    let map: Vec<i32> = (0..4).collect();
    let sources = vec![FlatVectorMergeSource {
        values: MergeSourceValues::Float32(src.float_vector_values(0).unwrap()),
        doc_map: &map,
    }];

    let mut writer = FlatVectorsWriter::new(4, &ID, "");
    writer
        .merge_one_flat_vector_field(&MergedFlatVectorField {
            field_number: 0,
            encoding: VectorEncoding::Float32,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            sources: &sources,
        })
        .unwrap();
    writer
        .write_field(&FlatVectorsField {
            field_number: 1,
            similarity: VectorSimilarityFunction::DotProduct,
            dimension: 3,
            docs: vec![1, 3],
            values: FieldVectorData::Byte(vec![1, 2, 3, 250, 251, 252]),
        })
        .unwrap();
    let (data, meta) = writer.finish();
    let reader = FlatVectorsReader::open(&meta, &data, &ID, "").unwrap();
    assert_eq!(reader.fields().len(), 2);
    assert_eq!(reader.float_vector_values(0).unwrap().size(), 4);
    let bytes = reader.byte_vector_values(1).unwrap();
    assert_eq!(bytes.vector(0).unwrap(), &[1, 2, 3]);
    assert_eq!(bytes.ord_to_doc(1).unwrap(), 3);
    // The merged field's region is still 64-byte aligned, and the byte
    // field's 4-byte aligned, exactly as a flush would leave them.
    assert_eq!(reader.field(0).unwrap().vector_data_offset % 4096, 0);
    assert_eq!(reader.field(1).unwrap().vector_data_offset % 4, 0);
}

#[test]
fn doc_to_ord_cursors_list_their_documents_in_bulk() {
    let words_of = |c: &DocToOrdCursor<'_>, n: usize| {
        let mut w = vec![0u64; n];
        c.or_docs_into(&mut w).unwrap();
        w
    };
    assert_eq!(words_of(&DocToOrdCursor::Empty, 2), vec![0, 0]);
    assert_eq!(words_of(&DocToOrdCursor::Dense { size: 3 }, 1), vec![0b111]);
    // Documents past the words are dropped, not written out of bounds.
    assert_eq!(
        words_of(&DocToOrdCursor::Dense { size: 70 }, 1),
        vec![u64::MAX]
    );
    let docs = [1, 64, 65, 200, -3];
    assert_eq!(
        words_of(&DocToOrdCursor::Explicit(&docs), 3),
        vec![0b10, 0b11, 0]
    );
    // An IndexedDISI with sparse, dense and all-documents blocks.
    let docs: Vec<i32> = (0..140_000)
        .filter(|d| d % 3 == 0 || (5_000..10_000).contains(d) || (65_536..131_072).contains(d))
        .collect();
    let (bytes, jumps) = crate::indexed_disi::write_with_dense_rank_power(
        &docs,
        crate::indexed_disi::DEFAULT_DENSE_RANK_POWER,
    );
    let c = DocToOrdCursor::Sparse(Box::new(DisiCursor::new(
        &bytes,
        crate::indexed_disi::DEFAULT_DENSE_RANK_POWER,
        jumps,
    )));
    let w = words_of(&c, 140_000 / 64 + 1);
    let got: Vec<i32> = (0..140_000usize)
        .filter(|&d| (w[d >> 6] >> (d & 63)) & 1 == 1)
        .map(|d| d as i32)
        .collect();
    assert_eq!(got, docs);
    // Too few words for the set is corruption.
    let mut short = vec![0u64; 4];
    assert!(c.or_docs_into(&mut short).is_err());
}
